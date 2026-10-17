"""
RVC VITS Synthesizer and Neural Source Filter (NSF) Acoustic Generator Network.

Adapted from MIT-licensed Applio and RVC-Project source code:
- Upstream: https://github.com/IAHispano/Applio (rvc/lib/infer_pack/models.py)
- Upstream: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI (infer/lib/infer_pack/models.py)

MIT License
Copyright (c) 2023-2026 AI Hispano / RVC-Project

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

import math
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    """
    Initialize module convolutional weights using a normal distribution.

    Args:
        m (nn.Module): Target PyTorch module.
        mean (float): Gaussian distribution mean. Defaults to 0.0.
        std (float): Gaussian distribution standard deviation. Defaults to 0.01.

    Returns:
        None

    Raises:
        None
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and hasattr(m, "weight") and m.weight is not None:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """
    Calculate symmetric padding length for causal or non-causal 1D convolutions.

    Args:
        kernel_size (int): Size of the convolutional kernel.
        dilation (int): Dilation factor of the kernel. Defaults to 1.

    Returns:
        int: Total padding required to preserve sequence length.

    Raises:
        None
    """
    return int((kernel_size * dilation - dilation) / 2)


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Create a boolean mask tensor indicating valid sequence elements.

    Args:
        lengths (torch.Tensor): 1D tensor containing lengths of sequences in batch.
        max_len (Optional[int]): Upper bound on sequence length. Defaults to max(lengths).

    Returns:
        torch.Tensor: Boolean mask of shape [batch_size, max_len].

    Raises:
        None
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    x = torch.arange(max_len, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


class LayerNorm(nn.Module):
    """
    Channel-first layer normalization module for 1D acoustic feature tensors.
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        """
        Initialize learnable scale and bias parameters.

        Args:
            channels (int): Dimensionality of input feature vectors.
            eps (float): Small constant for numerical stability. Defaults to 1e-5.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute channel-first layer normalization.

        Args:
            x (torch.Tensor): Feature tensor [batch, channels, time].

        Returns:
            torch.Tensor: Normalized feature tensor [batch, channels, time].

        Raises:
            None
        """
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module with optional relative position embeddings.
    """

    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize multi-head attention projection weights.

        Args:
            channels (int): Input channel dimension.
            out_channels (int): Output projected channel dimension.
            n_heads (int): Number of parallel attention heads.
            p_dropout (float): Dropout probability. Defaults to 0.0.
            window_size (Optional[int]): Relative position local attention window size.
            **kwargs: Unused arguments for backward compatibility.

        Returns:
            None

        Raises:
            AssertionError: If channels is not divisible by n_heads.
        """
        super().__init__()
        assert channels % n_heads == 0, f"channels {channels} must be divisible by n_heads {n_heads}"
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.k_channels = channels // n_heads

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_rel_pos = 2 * window_size + 1
            rel_std = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(1, n_rel_pos, self.k_channels) * rel_std)
            self.emb_rel_v = nn.Parameter(torch.randn(1, n_rel_pos, self.k_channels) * rel_std)

        init_weights(self)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-head attention map and projected representations.

        Args:
            x (torch.Tensor): Query tensor [batch, channels, time_q].
            c (torch.Tensor): Key/Value tensor [batch, channels, time_k].
            attn_mask (Optional[torch.Tensor]): Mask tensor [batch, 1, time_q, time_k].

        Returns:
            torch.Tensor: Attended and projected tensor [batch, out_channels, time_q].

        Raises:
            None
        """
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        b, _, t_s = q.size()
        t_t = k.size(2)

        q = q.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        k = k.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        v = v.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.k_channels)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e4)

        if self.window_size is not None and t_s == t_t:
            block_mask = torch.ones_like(scores).triu(-self.window_size).tril(self.window_size)
            scores = scores.masked_fill(block_mask == 0, -1e4)

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, v)
        output = output.transpose(2, 3).contiguous().view(b, self.n_heads * self.k_channels, t_s)
        return self.conv_o(output)


class FFN(nn.Module):
    """
    Position-wise feed-forward network with 1D dilated convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
    ) -> None:
        """
        Initialize position-wise 1D convolutional layers.

        Args:
            in_channels (int): Input representation dimension.
            out_channels (int): Output feature dimension.
            filter_channels (int): Intermediate expanded dimension.
            kernel_size (int): Convolution kernel size.
            p_dropout (float): Dropout probability. Defaults to 0.0.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)
        init_weights(self)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Execute feed-forward convolutions with ReLU activation and residual masking.

        Args:
            x (torch.Tensor): Input representation [batch, in_channels, time].
            x_mask (torch.Tensor): Binary sequence mask [batch, 1, time].

        Returns:
            torch.Tensor: Filtered representation [batch, out_channels, time].

        Raises:
            None
        """
        h = self.conv_1(x * x_mask)
        h = torch.relu(h)
        h = self.drop(h)
        h = self.conv_2(h * x_mask)
        return h * x_mask


class Encoder(nn.Module):
    """
    Multi-layer Transformer encoder utilizing MultiHeadAttention and FFN blocks.
    """

    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 10,
    ) -> None:
        """
        Initialize encoder attention and FFN layer blocks.

        Args:
            hidden_channels (int): Hidden dimension size.
            filter_channels (int): FFN expansion size.
            n_heads (int): Number of attention heads.
            n_layers (int): Number of transformer layers.
            kernel_size (int): FFN kernel size. Defaults to 1.
            p_dropout (float): Dropout probability. Defaults to 0.0.
            window_size (int): Attention window size. Defaults to 10.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()

        for _ in range(n_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size)
            )
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout)
            )
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through transformer encoder blocks.

        Args:
            x (torch.Tensor): Feature representations [batch, hidden_channels, time].
            x_mask (torch.Tensor): Binary sequence mask [batch, 1, time].

        Returns:
            torch.Tensor: Transformed representation [batch, hidden_channels, time].

        Raises:
            None
        """
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            x = self.norm_layers_2[i](x + y)
        return x * x_mask


class TextEncoder(nn.Module):
    """
    RVC Acoustic text/content token encoder with fundamental frequency pitch guidance.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        f0: bool = True,
    ) -> None:
        """
        Initialize content token linear projections and transformer encoder.

        Args:
            in_channels (int): Input acoustic feature channels (256 for v1, 768 for v2).
            out_channels (int): Output latent flow channel dimension.
            hidden_channels (int): Encoder hidden dimension.
            filter_channels (int): FFN expansion dimension.
            n_heads (int): Number of multi-head attention heads.
            n_layers (int): Number of transformer encoder layers.
            kernel_size (int): FFN convolution kernel size.
            p_dropout (float): Dropout probability.
            f0 (bool): Whether F0 pitch embeddings are enabled. Defaults to True.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.f0 = f0

        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(LRELU_SLOPE)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)

        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        phone: torch.Tensor,
        pitch: Optional[torch.Tensor],
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode content representations and pitch tokens into latent prior distributions.

        Args:
            phone (torch.Tensor): Input content embeddings [batch, time, in_channels].
            pitch (Optional[torch.Tensor]): Quantized pitch tokens [batch, time].
            lengths (torch.Tensor): Length of valid elements per batch element [batch].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Latent mean, log-variance, and mask.

        Raises:
            None
        """
        if self.f0 and pitch is not None:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        else:
            x = self.emb_phone(phone)
        x = x * math.sqrt(self.hidden_channels)
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class ResBlock1(nn.Module):
    """
    Residual Block Type 1 with dilated convolutions for HiFi-GAN generators.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, ...] = (1, 3, 5)) -> None:
        """
        Initialize stacked dilated 1D convolutions.

        Args:
            channels (int): Input and output channel dimension.
            kernel_size (int): Convolution kernel size. Defaults to 3.
            dilation (Tuple[int, ...]): Dilation rates per layer. Defaults to (1, 3, 5).

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilation:
            self.convs1.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            )
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections and LeakyReLU activations.

        Args:
            x (torch.Tensor): Input representation [batch, channels, time].

        Returns:
            torch.Tensor: Filtered representation [batch, channels, time].

        Raises:
            None
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization hooks.

        Returns:
            None

        Raises:
            None
        """
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class ResBlock2(nn.Module):
    """
    Residual Block Type 2 with dilated convolutions for HiFi-GAN generators.
    """

    def __init__(self, channels: int, kernel_size: int = 3, dilation: Tuple[int, ...] = (1, 3)) -> None:
        """
        Initialize residual block convolutions.

        Args:
            channels (int): Channel dimension.
            kernel_size (int): Convolution kernel size. Defaults to 3.
            dilation (Tuple[int, ...]): Dilation rates per layer. Defaults to (1, 3).

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilation:
            self.convs.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            )
        self.convs.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Filtered output tensor.

        Raises:
            None
        """
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization hooks.

        Returns:
            None

        Raises:
            None
        """
        for c in self.convs:
            remove_weight_norm(c)


class SineGen(nn.Module):
    """
    Harmonic sinusoidal waveform generator for Neural Source Filter excitation.
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
    ) -> None:
        """
        Initialize sine generator parameters.

        Args:
            samp_rate (int): Sampling rate in Hz.
            harmonic_num (int): Number of harmonic overtones. Defaults to 0.
            sine_amp (float): Amplitude of sine waveform. Defaults to 0.1.
            noise_std (float): Standard deviation of additive Gaussian noise. Defaults to 0.003.
            voiced_threshold (float): F0 threshold for voiced classification. Defaults to 0.0.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def forward(self, f0: torch.Tensor, upp: int) -> Tuple[torch.Tensor, torch.Tensor, None]:
        """
        Synthesize sinusoidal harmonic source signal from continuous F0 track.

        Args:
            f0 (torch.Tensor): Fundamental frequency contour [batch, time].
            upp (int): Upsampling factor to reach output audio sample rate.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, None]: Generated sine waves, voiced mask, and None.

        Raises:
            None
        """
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in range(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            rad_values = (f0_buf / float(self.sampling_rate)) % 1
            rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=float(upp), mode="linear").transpose(1, 2)

            rand_ini = torch.rand(f0.shape[0], self.dim, device=f0.device)
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

            tmp_over_one = torch.cumsum(rad_values, 1)
            tmp_over_one *= 2 * np.pi
            sine_waves = torch.sin(tmp_over_one)

            uv = torch.ones_like(f0) * (f0 > self.voiced_threshold)
            uv = F.interpolate(uv.transpose(1, 2), scale_factor=float(upp), mode="nearest").transpose(1, 2)

            noise_amp = uv * self.noise_std + (1.0 - uv) * self.sine_amp / 3.0
            noise = noise_amp * torch.randn_like(sine_waves)

            sine_waves = sine_waves * uv + noise
            return sine_waves, uv, None


class SourceModuleHnNSF(nn.Module):
    """
    Harmonic-Noise Source Module for Neural Source Filter (HN-NSF) vocoders.
    """

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0.0,
        is_half: bool = False,
    ) -> None:
        """
        Initialize HN-NSF source generator and projection layers.

        Args:
            sampling_rate (int): Output audio sample rate in Hz.
            harmonic_num (int): Number of harmonic overtones. Defaults to 0.
            sine_amp (float): Sine wave amplitude. Defaults to 0.1.
            add_noise_std (float): Additive noise standard deviation. Defaults to 0.003.
            voiced_threshold (float): Voicing threshold. Defaults to 0.0.
            is_half (bool): Whether half precision is enabled. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshold
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, upp: Optional[int] = None) -> Tuple[torch.Tensor, None, None]:
        """
        Generate excitation source signal from F0 pitch trajectory.

        Args:
            x (torch.Tensor): Continuous F0 pitch contour tensor [batch, time].
            upp (Optional[int]): Audio upsampling scale factor.

        Returns:
            Tuple[torch.Tensor, None, None]: Merged excitation source waveform.

        Raises:
            None
        """
        sine_wavs, _, _ = self.l_sin_gen(x, upp if upp is not None else 1)
        if self.is_half:
            sine_wavs = sine_wavs.half()
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None


class GeneratorNSF(nn.Module):
    """
    Neural Source Filter (NSF) HiFi-GAN acoustic generator driven by pitch excitation.
    """

    def __init__(
        self,
        initial_channel: int,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        gin_channels: int,
        sr: int,
        is_half: bool = False,
    ) -> None:
        """
        Initialize NSF-HiFi-GAN vocoder upsampling layers and residual blocks.

        Args:
            initial_channel (int): Input latent channel dimension.
            resblock (str): Residual block architecture ("1" or "2").
            resblock_kernel_sizes (List[int]): Convolution kernel sizes for residual blocks.
            resblock_dilation_sizes (List[List[int]]): Dilation factors for residual blocks.
            upsample_rates (List[int]): Up-sampling strides for transposed convolutions.
            upsample_initial_channel (int): Channels in initial upsampler stage.
            upsample_kernel_sizes (List[int]): Kernel sizes for transposed convolutions.
            gin_channels (int): Global conditioning speaker embedding dimension.
            sr (int): Target sampling rate in Hz.
            is_half (bool): Whether FP16 precision is enabled. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upp = int(np.prod(upsample_rates))

        self.f0_upsamp = nn.Upsample(scale_factor=self.upp)
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr, harmonic_num=0, is_half=is_half
        )
        self.noise_convs = nn.ModuleList()
        self.conv_pre = nn.Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        ResBlockClass = ResBlock1 if resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_in = upsample_initial_channel // (2 ** i)
            c_out = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(c_in, c_out, k, u, padding=(k - u) // 2)
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = int(np.prod(upsample_rates[i + 1 :]))
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        c_out,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_out, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlockClass(ch, k, tuple(d)))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(
        self,
        x: torch.Tensor,
        f0: torch.Tensor,
        g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Synthesize audio waveform conditioned on latent features, pitch, and speaker embeddings.

        Args:
            x (torch.Tensor): Latent representation [batch, initial_channel, time].
            f0 (torch.Tensor): Continuous F0 pitch contour [batch, time].
            g (Optional[torch.Tensor]): Speaker conditioning tensor [batch, gin_channels, 1].

        Returns:
            torch.Tensor: Synthesized 1D audio waveform [batch, 1, audio_samples].

        Raises:
            None
        """
        har_source, _, _ = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)

        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x_source = self.noise_convs[i](har_source)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization hooks for fast inference evaluation.

        Returns:
            None

        Raises:
            None
        """
        for up in self.ups:
            try:
                remove_weight_norm(up)
            except Exception:
                pass
        for block in self.resblocks:
            try:
                block.remove_weight_norm()
            except Exception:
                pass


class WN(nn.Module):
    """
    WaveNet dilated convolution layer stack for normalizing flow coupling layers.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
        p_dropout: float = 0.0,
    ) -> None:
        """
        Initialize WaveNet layer stack.

        Args:
            hidden_channels (int): Channel dimension.
            kernel_size (int): Convolution kernel size.
            dilation_rate (int): Dilation exponent base.
            n_layers (int): Number of WaveNet layers.
            gin_channels (int): Global conditioning dimension. Defaults to 0.
            p_dropout (float): Dropout probability. Defaults to 0.0.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            self.cond_layer = weight_norm(nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1))

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            self.in_layers.append(
                weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding))
            )
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels
            self.res_skip_layers.append(weight_norm(nn.Conv1d(hidden_channels, res_skip_channels, 1)))

    def forward(
        self, x: torch.Tensor, x_mask: torch.Tensor, g: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Execute WaveNet dilated convolutions with gated activation units.

        Args:
            x (torch.Tensor): Feature tensor [batch, hidden_channels, time].
            x_mask (torch.Tensor): Binary sequence mask [batch, 1, time].
            g (Optional[torch.Tensor]): Global conditioning tensor.

        Returns:
            torch.Tensor: Filtered feature tensor [batch, hidden_channels, time].

        Raises:
            None
        """
        output = torch.zeros_like(x)
        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = x_in + g_l
            t_act = torch.tanh(acts[:, : self.hidden_channels, :])
            s_act = torch.sigmoid(acts[:, self.hidden_channels :, :])
            acts = t_act * s_act

            acts = self.drop(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization hooks.

        Returns:
            None

        Raises:
            None
        """
        if hasattr(self, "cond_layer"):
            remove_weight_norm(self.cond_layer)
        for layer in self.in_layers:
            remove_weight_norm(layer)
        for layer in self.res_skip_layers:
            remove_weight_norm(layer)


class Flip(nn.Module):
    """
    Normalizing flow channel flipping layer.
    """

    def forward(
        self,
        x: torch.Tensor,
        *args: Any,
        reverse: bool = False,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Invert channel dimension ordering.

        Args:
            x (torch.Tensor): Input representation.
            reverse (bool): Flow direction. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Flipped representation.

        Raises:
            None
        """
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
            return x, logdet
        return x

    def remove_weight_norm(self) -> None:
        """
        No-op weight normalization hook removal.

        Returns:
            None

        Raises:
            None
        """
        pass


class ResidualCouplingLayer(nn.Module):
    """
    Affine residual coupling layer parameterized by WaveNet.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        p_dropout: float = 0.0,
        gin_channels: int = 0,
        mean_only: bool = False,
    ) -> None:
        """
        Initialize affine coupling projections and WaveNet parameterizer.

        Args:
            channels (int): Input representation channels.
            hidden_channels (int): WaveNet hidden channels.
            kernel_size (int): Convolution kernel size.
            dilation_rate (int): Dilation growth rate.
            n_layers (int): Number of WaveNet layers.
            p_dropout (float): Dropout probability. Defaults to 0.0.
            gin_channels (int): Global conditioning channel dimension.
            mean_only (bool): Whether to parameterize mean shift only. Defaults to False.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = nn.Conv1d(
            hidden_channels,
            self.half_channels * (1 if mean_only else 2),
            1,
        )
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Execute forward or reverse affine coupling transformation.

        Args:
            x (torch.Tensor): Feature tensor [batch, channels, time].
            x_mask (torch.Tensor): Binary sequence mask [batch, 1, time].
            g (Optional[torch.Tensor]): Global speaker conditioning tensor.
            reverse (bool): Whether to perform inverse flow transformation. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Transformed feature tensor.

        Raises:
            None
        """
        x0, x1 = torch.split(x, [self.half_channels, self.half_channels], 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels, self.half_channels], 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization hooks.

        Returns:
            None

        Raises:
            None
        """
        self.enc.remove_weight_norm()


class ResidualCouplingBlock(nn.Module):
    """
    Stacked normalizing flow residual coupling blocks with alternating channel flips.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ) -> None:
        """
        Initialize sequence of affine coupling layers and channel inversions.

        Args:
            channels (int): Latent channel dimension.
            hidden_channels (int): WaveNet hidden dimension.
            kernel_size (int): Convolution kernel size.
            dilation_rate (int): Dilation exponent factor.
            n_layers (int): Number of layers per WaveNet.
            n_flows (int): Number of coupling blocks. Defaults to 4.
            gin_channels (int): Speaker embedding conditioning channels. Defaults to 0.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Transform latent representation through stacked coupling layers.

        Args:
            x (torch.Tensor): Feature representations [batch, channels, time].
            x_mask (torch.Tensor): Binary sequence mask [batch, 1, time].
            g (Optional[torch.Tensor]): Global conditioning tensor.
            reverse (bool): Invert flow direction. Defaults to False.

        Returns:
            torch.Tensor: Transformed representation [batch, channels, time].

        Raises:
            None
        """
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization hooks across all coupling layers.

        Returns:
            None

        Raises:
            None
        """
        for flow in self.flows:
            flow.remove_weight_norm()


class SynthesizerTrn(nn.Module):
    """
    RVC VITS Acoustic Synthesizer supporting both RVC v1 and RVC v2 NSF architectures.
    """

    def __init__(
        self,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: List[int],
        resblock_dilation_sizes: List[List[int]],
        upsample_rates: List[int],
        upsample_initial_channel: int,
        upsample_kernel_sizes: List[int],
        spk_embed_dim: int,
        gin_channels: int,
        sr: Union[int, str],
        is_half: bool = False,
        version: str = "v2",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the complete RVC SynthesizerTrn generator network.

        Args:
            spec_channels (int): Input spectral feature dimension.
            segment_size (int): Segment length during training.
            inter_channels (int): Latent flow channel dimension.
            hidden_channels (int): Text encoder hidden dimension.
            filter_channels (int): FFN filter expansion dimension.
            n_heads (int): Number of multi-head attention heads.
            n_layers (int): Number of transformer encoder layers.
            kernel_size (int): Encoder FFN convolution kernel size.
            p_dropout (float): Dropout probability.
            resblock (str): Residual block type ("1" or "2").
            resblock_kernel_sizes (List[int]): Kernel sizes for HiFi-GAN ResBlocks.
            resblock_dilation_sizes (List[List[int]]): Dilation factors for HiFi-GAN ResBlocks.
            upsample_rates (List[int]): Up-sampling strides for transposed convolutions.
            upsample_initial_channel (int): Number of initial channels for upsampler.
            upsample_kernel_sizes (List[int]): Kernel sizes for transposed convolutions.
            spk_embed_dim (int): Number of speaker embedding identities.
            gin_channels (int): Global conditioning channel dimension.
            sr (Union[int, str]): Target sampling rate (e.g. 40000 or "40k").
            is_half (bool): Whether FP16 precision is enabled. Defaults to False.
            version (str): Model version ("v1" or "v2"). Defaults to "v2".
            **kwargs: Additional architectural configurations.

        Returns:
            None

        Raises:
            None
        """
        super().__init__()
        if isinstance(sr, str):
            if sr.endswith("k"):
                sr = int(float(sr[:-1]) * 1000)
            else:
                sr = int(sr)

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.sr = sr
        self.is_half = is_half
        self.version = version

        # Input feature dimensionality: 256 for v1, 768 for v2
        in_channels = 256 if version == "v1" else 768

        self.enc_p = TextEncoder(
            in_channels,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            f0=True,
        )
        self.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            is_half=is_half,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            3,
            gin_channels=gin_channels,
        )
        self.emb_g = nn.Embedding(spk_embed_dim, gin_channels)

    def remove_weight_norm(self) -> None:
        """
        Remove weight normalization hooks for fast inference evaluation.

        Returns:
            None

        Raises:
            None
        """
        try:
            self.flow.remove_weight_norm()
        except Exception:
            pass
        try:
            self.dec.remove_weight_norm()
        except Exception:
            pass

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        max_len: Optional[int] = None,
        return_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Synthesize audio waveform from content tokens and pitch contours.

        Args:
            phone (torch.Tensor): Extracted ContentVec representations [1, time, in_channels].
            phone_lengths (torch.Tensor): Length of content sequence [1].
            pitch (torch.Tensor): Quantized F0 pitch tokens [1, time].
            nsff0 (torch.Tensor): Continuous F0 pitch tensor [1, time].
            sid (torch.Tensor): Target speaker ID [1].
            max_len (Optional[int]): Unused max length constraint.
            return_mask (bool): Whether to return the attention mask. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Synthesized 1D waveform [1, 1, samples] (and mask).

        Raises:
            None
        """
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec(z * x_mask, nsff0, g=g)
        if return_mask:
            return o, x_mask
        return o
