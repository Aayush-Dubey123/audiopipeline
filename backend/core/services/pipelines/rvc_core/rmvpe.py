"""
RVC RMVPE (Robust Model for Vocal Pitch Estimation) PyTorch Module.

Adapted from MIT-licensed Applio, RVC-Project, and Dream-High RMVPE source code:
- Upstream: https://github.com/IAHispano/Applio (rvc/lib/predictors/rmvpe.py)
- Upstream: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI (infer/lib/rmvpe.py)
- Research: https://github.com/Dream-High/RMVPE

MIT License
Copyright (c) 2023-2026 AI Hispano / RVC-Project / Dream-High

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel


class SecurityValidationError(Exception):
    """Raised when model checkpoint deserialization fails secure weights_only validation."""
    pass


class BiGRU(nn.Module):
    """
    Bidirectional Gated Recurrent Unit layer for pitch sequence temporal context.
    """

    def __init__(self, input_features: int, hidden_features: int, num_layers: int) -> None:
        """
        Initialize bidirectional GRU layers.

        Args:
            input_features (int): Dimensionality of input feature vectors.
            hidden_features (int): Hidden dimension size per direction.
            num_layers (int): Number of stacked recurrent layers.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward recurrent step over sequence.

        Args:
            x (torch.Tensor): Input sequence [batch, time, input_features].

        Returns:
            torch.Tensor: Bidirectional hidden sequence [batch, time, 2 * hidden_features].

        Raises:
            None
        """
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    """
    Two-layer residual 2D convolution block with batch normalization and ReLU activations.
    """

    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01) -> None:
        """
        Initialize residual block convolution layers.

        Args:
            in_channels (int): Input channel count.
            out_channels (int): Output channel count.
            momentum (float): Batch normalization momentum. Defaults to 0.01.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward residual 2D convolution pass.

        Args:
            x (torch.Tensor): 4D feature map [batch, in_channels, height, width].

        Returns:
            torch.Tensor: Convolved residual feature map [batch, out_channels, height, width].

        Raises:
            None
        """
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    """
    Encoder block with residual convolutions and optional 2D average pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Optional[Tuple[int, int]],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ) -> None:
        """
        Initialize encoder residual convolutional layers and pooling.

        Args:
            in_channels (int): Input channel count.
            out_channels (int): Output channel count.
            kernel_size (Optional[Tuple[int, int]]): 2D pooling kernel dimensions.
            n_blocks (int): Number of residual convolution blocks. Defaults to 1.
            momentum (float): Batch normalization momentum. Defaults to 0.01.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward encoder convolution and pooling pass.

        Args:
            x (torch.Tensor): Feature map [batch, in_channels, height, width].

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Convolved features,
                or tuple of (unpooled, pooled) features if kernel_size is specified.

        Raises:
            None
        """
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Intermediate(nn.Module):
    """
    Intermediate bottleneck encoder block sequence.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_inters: int,
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        """
        Initialize intermediate residual bottleneck blocks.

        Args:
            in_channels (int): Input channel count.
            out_channels (int): Bottleneck channel count.
            n_inters (int): Number of intermediate stages.
            n_blocks (int): Number of blocks per stage.
            momentum (float): Batch normalization momentum. Defaults to 0.01.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum))
        for _ in range(n_inters - 1):
            self.layers.append(ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward bottleneck processing pass.

        Args:
            x (torch.Tensor): Input feature map.

        Returns:
            torch.Tensor: Processed bottleneck feature map.

        Raises:
            None
        """
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    """
    Decoder block with transposed 2D convolution upsampling and skip-connection concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ) -> None:
        """
        Initialize transposed convolution and residual reconstruction layers.

        Args:
            in_channels (int): Input channel count.
            out_channels (int): Upsampled channel count.
            stride (Tuple[int, int]): Transposed convolution strides.
            n_blocks (int): Number of residual blocks. Defaults to 1.
            momentum (float): Batch normalization momentum. Defaults to 0.01.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x: torch.Tensor, concat_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward upsampling and skip-fusion pass.

        Args:
            x (torch.Tensor): Decoder input tensor [batch, in_channels, height, width].
            concat_tensor (torch.Tensor): Skip connection from encoder.

        Returns:
            torch.Tensor: Reconstructed feature map [batch, out_channels, height, width].

        Raises:
            None
        """
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class Encoder(nn.Module):
    """
    Deep U-Net encoder collecting multi-scale feature skip-tensors.
    """

    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: Tuple[int, int],
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ) -> None:
        """
        Initialize hierarchical encoder stages.

        Args:
            in_channels (int): Initial input channel count (1 for spectrogram).
            in_size (int): Height dimension of input spectrogram.
            n_encoders (int): Number of downsampling encoder stages.
            kernel_size (Tuple[int, int]): Pooling kernel sizes.
            n_blocks (int): Number of residual blocks per stage.
            out_channels (int): Initial stage output channels. Defaults to 16.
            momentum (float): Batch normalization momentum. Defaults to 0.01.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(self.n_encoders):
            self.layers.append(ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward hierarchical encoder pass.

        Args:
            x (torch.Tensor): Input spectrogram [batch, 1, time, freq].

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]: Bottleneck features and list of skip tensors.

        Raises:
            None
        """
        concat_tensors: List[torch.Tensor] = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Decoder(nn.Module):
    """
    Deep U-Net decoder reconstructing feature maps from skip connections.
    """

    def __init__(
        self,
        in_channels: int,
        n_decoders: int,
        stride: Tuple[int, int],
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        """
        Initialize hierarchical decoder stages.

        Args:
            in_channels (int): Bottleneck channel count.
            n_decoders (int): Number of upsampling decoder stages.
            stride (Tuple[int, int]): Transposed convolution strides.
            n_blocks (int): Number of residual blocks per stage.
            momentum (float): Batch normalization momentum. Defaults to 0.01.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward hierarchical decoder pass.

        Args:
            x (torch.Tensor): Bottleneck feature map.
            concat_tensors (List[torch.Tensor]): Encoder skip connections.

        Returns:
            torch.Tensor: Reconstructed feature map.

        Raises:
            None
        """
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    """
    Full Deep U-Net encoder-decoder network with intermediate bottleneck.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int],
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        """
        Initialize DeepUnet architecture.

        Args:
            kernel_size (Tuple[int, int]): Pooling and upsampling strides.
            n_blocks (int): Residual blocks per stage.
            en_de_layers (int): Number of encoder/decoder layers. Defaults to 5.
            inter_layers (int): Number of bottleneck layers. Defaults to 4.
            in_channels (int): Input spectrogram channels. Defaults to 1.
            en_out_channels (int): Initial encoder stage channels. Defaults to 16.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.encoder = Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward U-Net execution pass.

        Args:
            x (torch.Tensor): Input mel spectrogram tensor.

        Returns:
            torch.Tensor: U-Net filtered acoustic feature representation.

        Raises:
            None
        """
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    """
    End-to-End Deep U-Net and Bidirectional GRU pitch classification model.
    """

    def __init__(
        self,
        n_blocks: int = 4,
        n_gru: int = 1,
        kernel_size: Tuple[int, int] = (2, 2),
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        """
        Initialize RMVPE E2E neural network.

        Args:
            n_blocks (int): Number of residual blocks per stage. Defaults to 4.
            n_gru (int): Number of BiGRU layers. Defaults to 1.
            kernel_size (Tuple[int, int]): Convolution kernel sizes. Defaults to (2, 2).
            en_de_layers (int): Number of U-Net stages. Defaults to 5.
            inter_layers (int): Number of bottleneck stages. Defaults to 4.
            in_channels (int): Input channel count. Defaults to 1.
            en_out_channels (int): Initial U-Net channels. Defaults to 16.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, n_gru),
            nn.Linear(512, 360),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Compute pitch bin posterior probabilities across 360 cent bins.

        Args:
            mel (torch.Tensor): Mel spectrogram tensor [batch, time, 128].

        Returns:
            torch.Tensor: Pitch posterior probabilities [batch, time, 360].

        Raises:
            None
        """
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class MelSpectrogram(nn.Module):
    """
    Differentiable Mel-Spectrogram extractor utilizing Hann-windowed STFT.
    """

    def __init__(
        self,
        n_mel_channels: int = 128,
        sampling_rate: int = 16000,
        win_length: int = 1024,
        hop_length: int = 160,
        mel_fmin: float = 30.0,
        mel_fmax: float = 8000.0,
        clamp: float = 1e-5,
    ) -> None:
        """
        Initialize Mel-Spectrogram filterbank and window buffers.

        Args:
            n_mel_channels (int): Number of Mel frequency bands. Defaults to 128.
            sampling_rate (int): Audio sampling rate. Defaults to 16000.
            win_length (int): STFT window length. Defaults to 1024.
            hop_length (int): STFT frame hop length. Defaults to 160.
            mel_fmin (float): Minimum Mel frequency. Defaults to 30.0.
            mel_fmax (float): Maximum Mel frequency. Defaults to 8000.0.
            clamp (float): Minimum log clamp threshold. Defaults to 1e-5.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.n_fft = win_length
        self.win_length = win_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

        mel_basis = mel(
            sr=sampling_rate,
            n_fft=self.n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.register_buffer("hann_window", torch.hann_window(win_length))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert raw waveform to log-Mel spectrogram.

        Args:
            audio (torch.Tensor): Audio waveform tensor [batch, samples].

        Returns:
            torch.Tensor: Log-Mel spectrogram tensor [batch, n_mel_channels, frames].

        Raises:
            None
        """
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window,
            center=True,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class RMVPE:
    """
    Robust Model for Vocal Pitch Estimation (RMVPE) inference pipeline.
    """

    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        """
        Initialize and load RMVPE model using strict safe weights_only deserialization.

        Args:
            model_path (str): Filesystem path to rmvpe.pt checkpoint.
            device (Optional[str]): Device identifier ('cuda:0' or 'cpu'). Defaults to CUDA if available.

        Returns:
            None

        Raises:
            SecurityValidationError: If checkpoint deserialization fails weights_only validation.
            FileNotFoundError: If checkpoint file does not exist.
        """
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.mel_extractor = MelSpectrogram(128, 16000, 1024, 160, 30.0, 8000.0).to(self.device)
        self.model = E2E(4, 1, (2, 2))

        try:
            # Enforce strict weights_only deserialization
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(ckpt)
        except Exception as exc:
            raise SecurityValidationError(
                f"RMVPE checkpoint at '{model_path}' failed safe weights_only validation: {exc}"
            ) from exc

        self.model.eval()
        self.model = self.model.to(self.device)

        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Feed mel-spectrogram through E2E pitch network.

        Args:
            mel (torch.Tensor): Log-Mel spectrogram tensor.

        Returns:
            torch.Tensor: Pitch bin probability tensor.

        Raises:
            None
        """
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden: np.ndarray, thred: float = 0.03) -> np.ndarray:
        """
        Decode pitch bin probability distributions into fundamental frequencies (Hz).

        Args:
            hidden (np.ndarray): Probability array [frames, 360].
            thred (float): Pitch salience threshold. Defaults to 0.03.

        Returns:
            np.ndarray: Continuous F0 frequencies in Hz [frames].

        Raises:
            None
        """
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def to_local_average_cents(self, salience: np.ndarray, thred: float = 0.05) -> np.ndarray:
        """
        Compute weighted local average cent estimation around the probability peak.

        Args:
            salience (np.ndarray): Posterior pitch bin salience [frames, 360].
            thred (float): Threshold to filter unvoiced frames. Defaults to 0.05.

        Returns:
            np.ndarray: Predicted cent pitch contour [frames].

        Raises:
            None
        """
        center = np.argmax(salience, axis=1)
        salience_padded = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience_padded[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        todo_salience_arr = np.array(todo_salience)
        todo_cents_mapping_arr = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience_arr * todo_cents_mapping_arr, 1)
        weight_sum = np.sum(todo_salience_arr, 1) + 1e-8
        devided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided

    def infer_from_audio(self, audio: Union[np.ndarray, torch.Tensor], thred: float = 0.03) -> np.ndarray:
        """
        Extract fundamental frequency contour from 16kHz audio input.

        Args:
            audio (Union[np.ndarray, torch.Tensor]): 1D audio sample array at 16000 Hz.
            thred (float): Unvoiced threshold. Defaults to 0.03.

        Returns:
            np.ndarray: 1D array of F0 values in Hz.

        Raises:
            None
        """
        if not torch.is_tensor(audio):
            audio_t = torch.from_numpy(audio)
        else:
            audio_t = audio

        mel_spec = self.mel_extractor(audio_t.float().to(self.device).unsqueeze(0))
        hidden = self.mel2hidden(mel_spec)
        hidden_np = hidden.squeeze(0).cpu().numpy()
        return self.decode(hidden_np, thred=thred)
