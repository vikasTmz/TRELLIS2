from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.norm import GroupNorm32, ChannelLayerNorm32
from ..modules.spatial import pixel_shuffle_3d
from ..modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32


def norm_layer(norm_type: str, *args, **kwargs) -> nn.Module:
    """
    Return a normalization layer.
    """
    if norm_type == "group":
        return GroupNorm32(32, *args, **kwargs)
    elif norm_type == "layer":
        return ChannelLayerNorm32(*args, **kwargs)
    else:
        raise ValueError(f"Invalid norm type {norm_type}")


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.norm1 = norm_layer(norm_type, channels)
        self.norm2 = norm_layer(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = zero_module(
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        )
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = h + self.skip_connection(x)
        return h


class DownsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "avgpool"] = "conv",
    ):
        assert mode in ["conv", "avgpool"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels, 2, stride=2)
        elif mode == "avgpool":
            assert (
                in_channels == out_channels
            ), "Pooling mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            return self.conv(x)
        else:
            return F.avg_pool3d(x, 2)


class UpsampleBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mode: Literal["conv", "nearest"] = "conv",
    ):
        assert mode in ["conv", "nearest"], f"Invalid mode {mode}"

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if mode == "conv":
            self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)
        elif mode == "nearest":
            assert (
                in_channels == out_channels
            ), "Nearest mode requires in_channels to be equal to out_channels"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv"):
            x = self.conv(x)
            return pixel_shuffle_3d(x, 2)
        else:
            return F.interpolate(x, scale_factor=2, mode="nearest")


class SparseStructureEncoder(nn.Module):
    """
    Encoder for Sparse Structure (\mathcal{E}_S in the paper Sec. 3.3).

    Args:
        in_channels (int): Channels of the input.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the encoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """

    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(in_channels, channels[0], 3, padding=1)

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(DownsampleBlock3d(ch, channels[i + 1]))

        self.middle_block = nn.Sequential(
            *[
                ResBlock3d(channels[-1], channels[-1])
                for _ in range(num_res_blocks_middle)
            ]
        )

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], latent_channels * 2, 3, padding=1),
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(
        self, x: torch.Tensor, sample_posterior: bool = False, return_raw: bool = False
    ) -> torch.Tensor:
        h = self.input_layer(x)
        h = h.type(self.dtype)

        for block in self.blocks:
            h = block(h)
        h = self.middle_block(h)

        h = h.type(x.dtype)
        h = self.out_layer(h)

        mean, logvar = h.chunk(2, dim=1)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        if return_raw:
            return z, mean, logvar
        return z


class SparseStructureDecoder(nn.Module):
    """
    Decoder for Sparse Structure (\mathcal{D}_S in the paper Sec. 3.3).

    Args:
        out_channels (int): Channels of the output.
        latent_channels (int): Channels of the latent representation.
        num_res_blocks (int): Number of residual blocks at each resolution.
        channels (List[int]): Channels of the decoder blocks.
        num_res_blocks_middle (int): Number of residual blocks in the middle.
        norm_type (Literal["group", "layer"]): Type of normalization layer.
        use_fp16 (bool): Whether to use FP16.
    """

    def __init__(
        self,
        out_channels: int,
        latent_channels: int,
        num_res_blocks: int,
        channels: List[int],
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)

        self.middle_block = nn.Sequential(
            *[
                ResBlock3d(channels[0], channels[0])
                for _ in range(num_res_blocks_middle)
            ]
        )

        self.blocks = nn.ModuleList([])
        for i, ch in enumerate(channels):
            self.blocks.extend([ResBlock3d(ch, ch) for _ in range(num_res_blocks)])
            if i < len(channels) - 1:
                self.blocks.append(UpsampleBlock3d(ch, channels[i + 1]))

        self.out_layer = nn.Sequential(
            norm_layer(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )

        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.use_fp16 = True
        self.dtype = torch.float16
        self.blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.use_fp16 = False
        self.dtype = torch.float32
        self.blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)

        h = h.type(self.dtype)

        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)

        h = h.type(x.dtype)
        h = self.out_layer(h)
        return h


"""
Loaded latents: feats shape torch.Size([2606, 32]), coords shape torch.Size([2606, 4])
Decoding sparse structure coordinates...

Loaded latents: z_s shape torch.Size([8, 16, 16, 16]) 
2026-04-13 19:00:58,625 - INFO - Decoding sparse structure with model: 

SparseStructureDecoder(
  (input_layer): Conv3d(8, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (middle_block): Sequential(
    (0): ResBlock3d(
      (norm1): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
    (1): ResBlock3d(
      (norm1): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
  )
  (blocks): ModuleList(
    (0-1): 2 x ResBlock3d(
      (norm1): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
    (2): UpsampleBlock3d(
      (conv): Conv3d(512, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    )
    (3-4): 2 x ResBlock3d(
      (norm1): ChannelLayerNorm32((128,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((128,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
    (5): UpsampleBlock3d(
      (conv): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    )
    (6-7): 2 x ResBlock3d(
      (norm1): ChannelLayerNorm32((32,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((32,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
  )
  (out_layer): Sequential(
    (0): ChannelLayerNorm32((32,), eps=1e-05, elementwise_affine=True)
    (1): SiLU()
    (2): Conv3d(32, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  )
)

2026-04-13 19:00:58,847 - INFO - Decoded sparse structure shape before max pool: torch.Size([1, 1, 64, 64, 64])
2026-04-13 19:00:58,860 - INFO - Decoded sparse structure coordinates shape: torch.Size([2606, 4])


2026-04-13 19:00:58,869 - INFO - Decoding sparse structure with model: SparseStructureDecoder(
  (input_layer): Conv3d(8, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (middle_block): Sequential(
    (0): ResBlock3d(
      (norm1): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
    (1): ResBlock3d(
      (norm1): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
  )
  (blocks): ModuleList(
    (0-1): 2 x ResBlock3d(
      (norm1): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
    (2): UpsampleBlock3d(
      (conv): Conv3d(512, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    )
    (3-4): 2 x ResBlock3d(
      (norm1): ChannelLayerNorm32((128,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((128,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
    (5): UpsampleBlock3d(
      (conv): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    )
    (6-7): 2 x ResBlock3d(
      (norm1): ChannelLayerNorm32((32,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((32,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
  )
  (out_layer): Sequential(
    (0): ChannelLayerNorm32((32,), eps=1e-05, elementwise_affine=True)
    (1): SiLU()
    (2): Conv3d(32, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  )
)

2026-04-13 19:00:58,879 - INFO - Decoded sparse structure shape before max pool: torch.Size([1, 1, 64, 64, 64])
2026-04-13 19:00:58,890 - INFO - Decoded sparse structure coordinates shape: torch.Size([991, 4])


2026-04-13 19:00:58,897 - INFO - Decoding sparse structure with model: SparseStructureDecoder(
  (input_layer): Conv3d(8, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  (middle_block): Sequential(
    (0): ResBlock3d(
      (norm1): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
    (1): ResBlock3d(
      (norm1): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
  )
  (blocks): ModuleList(
    (0-1): 2 x ResBlock3d(
      (norm1): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((512,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
    (2): UpsampleBlock3d(
      (conv): Conv3d(512, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    )
    (3-4): 2 x ResBlock3d(
      (norm1): ChannelLayerNorm32((128,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((128,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
    (5): UpsampleBlock3d(
      (conv): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
    )
    (6-7): 2 x ResBlock3d(
      (norm1): ChannelLayerNorm32((32,), eps=1e-05, elementwise_affine=True)
      (norm2): ChannelLayerNorm32((32,), eps=1e-05, elementwise_affine=True)
      (conv1): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (conv2): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
      (skip_connection): Identity()
    )
  )
  (out_layer): Sequential(
    (0): ChannelLayerNorm32((32,), eps=1e-05, elementwise_affine=True)
    (1): SiLU()
    (2): Conv3d(32, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
  )
)

Decoded sparse structure shape before max pool: torch.Size([1, 1, 64, 64, 64])
Decoded sparse structure coordinates shape: torch.Size([361, 4])


"""
