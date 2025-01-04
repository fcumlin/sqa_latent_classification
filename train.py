import argparse
import logging
import os
import shutil
from typing import Any, Type

import gin
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.model_selection
import sklearn.neighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import tqdm

from dataset import LibriAugmented, get_dataloader
from model import Dnsmos, DnsmosClassic
import test


parser = argparse.ArgumentParser(description='Gin and save path.')
parser.add_argument(
    '--gin_path',
    type=str,
    help='Path to the gin-config.',
    default='configs/tot.gin'
)
parser.add_argument(
    '--save_path',
    type=str,
    help='Path to directory storing results.',
)
args = parser.parse_args()


@gin.configurable
class TrainingLoop:
    """The training loop which trains and evaluates a model."""

    def __init__(
        self,
        *,
        model: nn.Module = Dnsmos,
        save_path: str = '',
        loss_type: str = 'mse',
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        weight_decay: float = 0.0,
        dataset_cls: Type[Dataset] = LibriAugmented,
        num_epochs: int = 500,
        learning_rate: float = 1e-4,
        batch_size_train: int = 32,
        num_latent_features: int = 64,
    ):
        """Initializes the instance.
        
        Args:
            model: The nn.Module model. Expected to take (B, 1, T, F) as input,
                and output (B, S), where B is the batch size, T is the time bins,
                F is the frequency bins, and S is the score. Has to contain a
                method `encoder` which returns a latent representation, not
                necessarily one dimensional (`flatten` is applied).
            save_path: Path to log directory.
            loss_type: Type of loss, 'mse' or 'mae' are supported.
            optimizer: The optimizer.
            weight_decay: Weight decay of the parameters.
            dataset_cls: The dataset class. Expected to have the parameter
                `valid`, which can take the values 'train', 'val', and 'test'.
                Returns a `Dataset` object.
            num_epochs: Number of training epochs.
            learning_rate: The learning rate.
            batch_size_train: Batch size of train set.
            num_latent_features: The number of latent features of the model.
        """
        # Setup logging and paths.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self._save_path = save_path
        log_path = os.path.join(save_path, 'train.log')
        logging.basicConfig(filename=log_path, level=logging.INFO)

        # Datasets.
        train_dataset = dataset_cls(valid='train')
        valid_dataset = dataset_cls(valid='val')
        test_dataset = dataset_cls(valid='test')
        logging.info(f'Num train speech clips: {len(train_dataset)}')
        logging.info(f'Num val speech clips: {len(valid_dataset)}')
        logging.info(f'Num test speech clips: {len(test_dataset)}')
        self._train_loader = get_dataloader(
            dataset=train_dataset,
            batch_size=batch_size_train
        )
        self._valid_loader = get_dataloader(
            dataset=valid_dataset,
            batch_size=1
        )
        self._test_loader = get_dataloader(
            dataset=test_dataset,
            batch_size=1
        )
        self._esc50_dataset = test.ESC50()
        self._label_type = train_dataset.label_type

        # Model and optimizers.
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'Device={self._device}')
        self._model = model().to(self._device)
        self._num_latent_features = num_latent_features
        # TODO: Explore some learning rate scheduler.
        self._optimizer = optimizer(self._model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self._optimizer.zero_grad()
        if loss_type == 'mse':
            self._loss_fn = F.mse_loss
        elif loss_type == 'mae':
            self._loss_fn = F.l1_loss
        else:
            raise ValueError(f'Loss {loss_type} not supported.')
        self._all_loss = []
        self._epoch = 0
        self._num_epochs = num_epochs
        
        # ESC50 stored results.
        self._esc50_top1_acc = []
        self._esc50_top3_acc = []
    
    @property
    def save_path(self):
        """Returns the path to the log directory."""
        return self._save_path

    def _knn_on_augmentations(
        self,
        features: np.ndarray,
        z: np.ndarray,
        prefix: str,
        test_size: float = 0.3,
        n_neighbors: int = 15,
    ):
        """Returns kNN results on a splitted test data of features."""
        X = np.array(features)
        X_train, X_val, z_train, z_val = sklearn.model_selection.train_test_split(
            X, z, test_size=test_size, random_state=0, stratify=z)
        kNN_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        kNN_model.fit(X_train, np.argmax(z_train, axis=1))
        result = kNN_model.score(X_val, np.argmax(z_val, axis=1))
        logging.info(f"Augmentation kNN accuracy on {prefix}: {result}")
        return result
    
    def _knn_on_esc50(
        self,
        prefix: str,
        test_size: float = 0.3,
        n_neighbors: int = 15
    ) -> tuple[float, float]:
        """Returns top-1 and top-3 accuracy on ESC-50 dataset."""
        logging.info("ESC50 accuracy")
        X_train, X_val, z_train, z_val = self._esc50_dataset.get_latent_representations(
            self._momentum_model, self._num_latent_features, test_size, self._device)
        top_1, top_3 = self._esc50_dataset.knn(
            X_train=X_train,
            z_train=z_train,
            X_val=X_val,
            z_val=z_val,
            n_neighbors=n_neighbors,
            save_confusion_matrix=True if prefix != 'Valid' else False,
            save_dir=self._save_path,
        )
        logging.info(f'kNN accuracy: top_1={top_1 * 100:.2f} %, top_3={top_3 * 100:.2f} %')
        self._esc50_top1_acc.append(top_1)
        self._esc50_top3_acc.append(top_3)
        return top_1, top_3

    def log_esc50_results(self):
        """Logs the top-1 and top-3 accuracy on the ESC-50 dataset."""
        logging.info(self._esc50_top1_acc)
        logging.info(self._esc50_top3_acc)
            
    def _train_once(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        """Performs forward and backward pass on batch.
        
        Args:
            batch: The batch consisting of the spectrograms,
                labels and augmentations, in that order.
        """
        specs, labels, augmentations = batch
        specs = specs.to(self._device)
        specs = specs.unsqueeze(1)
        labels = labels.to(self._device)
        augmentations = augmentations.to(self._device)

        # Forward.
        predictions = self._model(specs)
        predictions = predictions.squeeze(-1)

        # Loss.
        loss = self._loss_fn(labels, predictions)
                
        # Backward.
        loss.backward()
        self._all_loss.append(loss.item())
        del loss

        # Gradient clipping.
        nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5)

        # Step.
        self._optimizer.step()
        self._optimizer.zero_grad()
    
    def train(self, valid_each_epoch: bool = True) -> None:
        """Trains the model on the train data `self._num_epochs` number of epochs.
        
        Args:
            valid_each_epoch: If to compute the validation performance.
        """
        self._model.train()
        while self._epoch <= self._num_epochs:
            self._all_loss = list()
            for batch in tqdm.tqdm(
                self._train_loader,
                ncols=0,
                desc="Train",
                unit=" step"
            ):
                self._train_once(batch)

            average_loss = torch.FloatTensor(self._all_loss).mean().item()
            logging.info(f'Average loss={average_loss}')

            if valid_each_epoch:
                self.valid()
            self._epoch += 1

    def _evaluate(self, dataloader: Any, prefix: str):
        """Evaluates the model on the data based on quality prediction and augmentation accuracy."""
        self._model.eval()
        predictions, labels = [], []
        num_examples = len(dataloader.dataset)
        features = np.zeros((num_examples, self._num_latent_features))
        z = np.zeros((num_examples, dataloader.dataset.num_augmentations))
        for i, batch in enumerate(tqdm.tqdm(
            dataloader,
            ncols=0,
            desc=prefix,
            unit=" step"
        )):
            wav, label, augmentation = batch
            wav = wav.to(self._device)
            wav = wav.unsqueeze(1) # shape (batch, 1, seq_len, [dim feature])

            with torch.no_grad():
                prediction = self._model(wav) # shape (batch, 1)
                prediction = prediction.squeeze(-1) # shape (batch)

                prediction = prediction.cpu().detach().numpy()
                predictions.extend(prediction.tolist())
                labels.extend(label.tolist())
                features[i, :] = self._model.encoder(
                    wav
                ).detach().cpu().numpy().flatten()
                z[i, augmentation.squeeze().numpy()] = 1

        predictions = np.array(predictions)
        labels = np.array(labels)
        utt_mse=np.mean((labels-predictions)**2)
        utt_pcc=np.corrcoef(labels, predictions)[0][1]
        utt_srcc=scipy.stats.spearmanr(labels, predictions)[0]

        logging.info(
            f"\n[{prefix}][{self._epoch}][UTT][ MSE = {utt_mse:.4f} | LCC = {utt_pcc:.4f} | SRCC = {utt_srcc:.4f} ]"
        )
        self._knn_on_augmentations(features, z, prefix)
        self._knn_on_esc50(prefix)
        self._model.train()
        return predictions, labels
    
    def valid(self):
        """Evaluates the model on validation data."""
        return self._evaluate(self._valid_loader, 'Valid')
    
    def test(self):
        """Evaluates the model on test data."""
        predictions, labels = self._evaluate(self._test_loader, 'Test')
        plt.scatter(labels, predictions)
        plt.xlim([0.8, 4.7])
        plt.ylim([0.8, 4.7])
        plt.xlabel(self._label_type)
        plt.ylabel('Predictions')
        plt.title('Test data predictions vs targets')
        plt.savefig(os.path.join(self._save_path, 'test_scatter.png'))
        return predictions, labels
    
    def save_model_old(self, model_name: str = 'model.pt'):
        """DEPRECATED: Saves the model."""
        torch.save(self._model, os.path.join(self._save_path, model_name))

    def save_model(self, model_name: str = 'model.pt'):
        """Saves the model."""
        model_scripted = torch.jit.script(self._model)
        model_scripted.save(os.path.join(self._save_path, model_name))


def main():
    """Main."""
    gin.external_configurable(
            torch.nn.modules.activation.ReLU,
            module='torch.nn.modules.activation'
            )
    gin.external_configurable(
            torch.nn.modules.activation.SiLU,
            module='torch.nn.modules.activation'
            )
    gin.parse_config_file(args.gin_path)

    train_loop = TrainingLoop(save_path=args.save_path)
    new_gin_path = os.path.join(train_loop.save_path, 'config.gin')
    shutil.copyfile(args.gin_path, new_gin_path)
    train_loop.train()
    train_loop.test()
    train_loop.log_esc50_results()
    train_loop.save_model()


if __name__ == '__main__':
    main()
