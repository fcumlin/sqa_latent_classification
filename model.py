from typing import Any, Union

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_conv_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int] = (3, 3),
    padding: tuple[int, int] = (1, 1),
    activation_fn: Any = nn.ReLU,
    max_pool_size: Union[tuple[int, int], int, None] = 3,
    dropout: float | None = 0.3,
    bn: bool = False,
) -> nn.Sequential:
    """Returns a CBAD layer: convolution, batch normalization, activation, and dropout."""
    layers = [nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding
    )]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(activation_fn())
    if max_pool_size is not None:
        layers.append(nn.MaxPool2d(max_pool_size))
    if dropout is not None:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# TODO: Consider adding an encoder config. 
@gin.configurable
class Encoder(nn.Module):
    
    def __init__(self, bn: bool = True, max_pool_size: int = 3, activation_fn: Any = nn.ReLU):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            _get_conv_layer(1, 32, bn=bn, max_pool_size=None, activation_fn=activation_fn),
            _get_conv_layer(32, 32, bn=bn, max_pool_size=max_pool_size, activation_fn=activation_fn),
            _get_conv_layer(32, 64, bn=bn, max_pool_size=None, activation_fn=activation_fn),
            _get_conv_layer(64, 64, bn=bn, max_pool_size=max_pool_size, activation_fn=activation_fn),
            _get_conv_layer(64, 128, max_pool_size=None, dropout=None, activation_fn=activation_fn),
        )

    def forward(self, speech_spectrum):
        # input speech_spectrum shape (batch, 1, max_seq_len, n_features)
        batch = speech_spectrum.shape[0]
        speech_spectrum = self.encoder(speech_spectrum) # shape (batch, 64, max_seq_len, n_features)
        embeddings = F.max_pool2d(speech_spectrum, kernel_size=speech_spectrum.size()[2:])
        embeddings = embeddings.view(batch, -1) # shape (batch, 64)
        return embeddings

    
@gin.configurable
def _dense_layer(
    in_dim: int,
    out_dim: int,
    use_ln: bool,
    use_activation: bool,
    activation_fn: Any = nn.ReLU,
) -> nn.Sequential:
    """Returns Sequential Dense-OptionalLayerNorm-OptionalActivation layer."""
    layers = [nn.Linear(in_dim, out_dim)]
    if use_ln:
        layers.append(nn.LayerNorm(out_dim))
    if use_activation:
        layers.append(activation_fn())
    return nn.Sequential(*layers)


@gin.configurable
class Head(nn.Module):

    def __init__(self, use_ln: bool = False, activation_fn: Any = nn.ReLU):
        super(Head, self).__init__()
        self.head = nn.Sequential(
            _dense_layer(128, 64, use_ln, True, activation_fn),
            _dense_layer(64, 64, use_ln, True, activation_fn),
            _dense_layer(64, 1, False, False),
        )

    def forward(self, embeddings):
        # input embeddings shape (batch, 64)
        prediction = self.head(embeddings)
        return prediction


@gin.configurable
class Dnsmos(nn.Module):
    
    def __init__(self):
        super(Dnsmos, self).__init__()
        self._encoder = Encoder()
        self._head = Head()

    def encoder(self, speech_spectrum):
        return self._encoder(speech_spectrum)

    def forward(self, speech_spectrum):
        embeddings = self._encoder(speech_spectrum)
        return self._head(embeddings)


@gin.configurable
class DnsmosEncoder(nn.Module):
    
    def __init__(self):
        super(DnsmosEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding=(1,1)),
            nn.ReLU(),
        )
        
    def forward(self, speech_spectrum):
        # Input speech_spectrum shape (batch, 1, max_seq_len, n_features).
        batch = speech_spectrum.shape[0]
        speech_spectrum = self.encoder(speech_spectrum) # shape (batch, 64, max_seq_len, n_features)
        embeddings = F.max_pool2d(speech_spectrum, kernel_size=speech_spectrum.size()[2:])
        embeddings = embeddings.view(batch, -1) # shape (batch, 64)
        return embeddings


@gin.configurable
class DnsmosHead(nn.Module):

    def __init__(self):
        super(DnsmosHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings):
        # input embeddings shape (batch, 64)
        prediction = self.head(embeddings)
        return prediction


@gin.configurable
class DnsmosClassic(nn.Module):
    
    def __init__(self):
        super(DnsmosClassic, self).__init__()
        self.encoder = DnsmosEncoder()
        self.head = DnsmosHead()

    def forward(self, speech_spectrum):
        embeddings = self.encoder(speech_spectrum)
        prediction = self.head(embeddings)
        return prediction
