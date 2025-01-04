"""Dataset loader for LibriSpeech."""

import os
from typing import Callable, Optional, Sequence, Union

import gin
import numpy as np
import librosa
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import tqdm

import audio
import utils


_AugmentationFn = Callable[[audio.Audio], audio.Audio]


@gin.configurable
class LibriAugmented(Dataset):
    """The LibriAugmented dataset."""
    
    def __init__(
        self,
        data_path: str = '../../datasets/LibriAugmented',
        valid: str = 'train',
        label_type: str = 'visqol',
        sample_rate: int = 16000,
        use_multi_augmentations: bool = False,
        num_samples_per_class: Optional[int] = None,
    ):
        """Initializes the instance.
        
        Args:
            data_path: Path to the dataset.
            valid: The data type. Can be 'train', 'val', or 'test'.
            label_type: The type of label. Can be 'pesq' or 'visqol'.
            sample_rate: The sample rate of the data. Note that the data is
                stored at 16 kHz; any larger rate would not give more information.
            use_multi_augmentations: If two use the double augmented dataset, which will
                double the amount of speech clips.
            num_sample_per_class: Specify to extract fewer samples. E.g., if 500 is given, 
                only 500 speech clips will be extracted per augmentation. There are expectedly
                at most 2836 speech clips per augmentation, using more might NOT make sense.
        """
        self._data_path = data_path
        self._df = pd.read_csv(os.path.join(data_path, f'{valid}.csv'))
        if use_multi_augmentations:
            tmp_df = pd.read_csv(os.path.join(data_path, f'{valid}2.csv'))
            self._df = pd.concat([self._df, tmp_df])
        self._num_samples = len(self._df)
        self._valid = valid
        self._label_type = label_type
        self._sample_rate = sample_rate

        self._num_samples_per_class = num_samples_per_class
        self._labels, self._augmentations, self._aug_map = self._load_labels()
        self._mag_specs = self._load_clips()

    @property
    def num_augmentations(self) -> int:
        """Returns number of augmentations in the dataset."""
        return len(self._aug_map)
    
    @property
    def label_type(self) -> str:
        """Returns the type of the label (i.e., 'pesq' or 'visqol')."""
        return self._label_type

    def _load_labels(self) -> tuple[list[float], list[int], dict[str, int]]:
        """Loads the labels."""
        labels, augmentations = [], []
        # Map augmentations to integers.
        aug_map = {}
        num_samples_extracted = {}
        for label, augmentation in tqdm.tqdm(
            zip(self._df[self._label_type], self._df['augmentation']),
            total=self._num_samples,
            desc='Loading labels...',
        ):
            if augmentation in num_samples_extracted:
                num_samples_extracted[augmentation] += 1
                if self._num_samples_per_class is not None and num_samples_extracted[augmentation] > self._num_samples_per_class:
                    continue
            else:
                num_samples_extracted[augmentation] = 1
            labels.append(label)
            if augmentation not in aug_map:
                aug_map[augmentation] = len(aug_map)
            augmentations.append(aug_map[augmentation])

        return labels, augmentations, aug_map

    def _load_clips(self) -> list[np.ndarray]:
        """Loads the clips and transforms to spectrograms"""
        mag_specs = []
        num_samples_extracted = {}
        for path, augmentation in tqdm.tqdm(
            zip(self._df['processed'], self._df['augmentation']),
            total=self._num_samples,
            desc='Loading clips...',
        ):
            if augmentation in num_samples_extracted:
                num_samples_extracted[augmentation] += 1
                if self._num_samples_per_class is not None and num_samples_extracted[augmentation] > self._num_samples_per_class:
                    continue
            else:
                num_samples_extracted[augmentation] = 1
            wav, _ = librosa.load(path, sr=self._sample_rate)
            signal = audio.Audio(wav, self._sample_rate)
            samples = np.squeeze(signal.samples)
            spec = utils.stft(samples)
            mag_specs.append(spec)

        return mag_specs
    
   
    def __getitem__(self, idx: int)  -> tuple[list[np.ndarray], list[float], list[int]]:
        """Returns a spectrogram with label and augmentation applied."""
        return self._mag_specs[idx], self._labels[idx], self._augmentations[idx]
   
    def __len__(self) -> int:
        """Returns the number of speech clips in the dataset."""
        return len(self._mag_specs)
 
    def collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a batch consisting of tensors."""
        mag_specs, labels, augmentations = zip(*batch)
        mag_specs = torch.FloatTensor(np.array(mag_specs))
        labels = torch.FloatTensor(labels)
        augmentations = torch.LongTensor(augmentations)
        return mag_specs, labels, augmentations


@gin.configurable
def get_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool
) -> DataLoader:
    """Returns a dataloader of the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn,
    )
