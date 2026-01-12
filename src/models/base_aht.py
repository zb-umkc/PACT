import torch
import torch.nn as nn
import numpy as np

from src.entropy_models.entropy_models import GGM
from src.utils.utils import _update_registered_buffer


class BB(nn.Module):

    def __init__(self, N: int, beta: float = 1.5):
        """
        Args:
            N: number of hyper channels (for z).
            beta: shape parameter of generalized Gaussian, usually 1.5.
        """
        super().__init__()

        self.N = int(N)

        # To be assigned in child classes (AHT model):
        self.g_a = None  # analysis transform
        self.g_s = None  # synthesis transform
        self.h_a = None  # hyper analysis (AHT encoder)
        self.h_s = None  # hyper synthesis (AHT decoder)

        # Hyperprior parameters for z (fully factorized)
        self.means_hyper = nn.Parameter(torch.zeros(1, N, 1, 1))
        self.scales_hyper = nn.Parameter(torch.ones(1, N, 1, 1))

        # Generalized Gaussian entropy model (β fixed as in the paper)
        # If GGM doesn't take beta in constructor in their code, just ignore arg
        self.entropy_estimation = GGM()  # beta handled inside GGM (β=1.5)

        self.init()

    # -------------------------------------------------------------
    # Buffers / CDF tables for arithmetic coding
    # -------------------------------------------------------------
    def init(self):
        # For hyper latents z
        self.register_buffer("scale_table", torch.Tensor())
        self.register_buffer("quantized_cdf_z", torch.Tensor())
        self.register_buffer("cdf_length_z", torch.Tensor())
        self.register_buffer("offset_z", torch.Tensor())
        # For main latents y
        self.register_buffer("quantized_cdf_y", torch.Tensor())
        self.register_buffer("cdf_length_y", torch.Tensor())
        self.register_buffer("offset_y", torch.Tensor())
        return True

    def update(self, scale_table: torch.Tensor):
        """
        Build / update CDF tables for z and y.

        scale_table: 1-D tensor of scales used to quantize GGM CDFs for y.
        """
        # Store scale table buffer
        self.register_buffer("scale_table", scale_table)

        # Hyperlatents z: use trainable scales_hyper
        q_cdf_z, cdf_len_z, offset_z = self.entropy_estimation.get_quantized_cdf(
            self.scales_hyper.detach().view(-1)
        )
        self.register_buffer("quantized_cdf_z", q_cdf_z)
        self.register_buffer("cdf_length_z", cdf_len_z)
        self.register_buffer("offset_z", offset_z)

        # Main latents y: use provided scale_table (vector of possible scales)
        q_cdf_y, cdf_len_y, offset_y = self.entropy_estimation.get_quantized_cdf(
            scale_table
        )
        self.register_buffer("quantized_cdf_y", q_cdf_y)
        self.register_buffer("cdf_length_y", cdf_len_y)
        self.register_buffer("offset_y", offset_y)
        return True

    def _update_registered_buffers(
        self,
        buffer_names,
        state_dict,
        policy: str = "resize_if_empty",
        dtype=torch.int,
    ):
        """
        Same helper as in the original repo: when loading a checkpoint,
        resize registered buffers according to the state_dict.
        """
        valid_buffer_names = [n for n, _ in self.named_buffers()]

        for buffer_name in buffer_names:
            if buffer_name not in valid_buffer_names:
                raise ValueError(f'Invalid buffer name "{buffer_name}"')

        for buffer_name in buffer_names:
            _update_registered_buffer(self, buffer_name, f"{buffer_name}", state_dict, policy, dtype)

    # -------------------------------------------------------------
    # Index builders for entropy coding
    # -------------------------------------------------------------
    @staticmethod
    def build_indexes_z(size):
        """
        Build index tensor for z with shape (N, C, H, W),
        just repeating channel indices over spatial dims.
        """
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    def build_indexes_conditional(self, scales: torch.Tensor):
        """
        Given per-pixel scales (B,C,H,W), map each scale to the closest
        entry in self.scale_table to build integer indexes.
        """
        device = scales.device
        # Last element of scale_table is typically sentinel; ignore it
        scale_table = self.scale_table[:-1].to(device).view(1, 1, 1, 1, -1)
        scales_expand = scales.unsqueeze(-1)
        indexes = (scales_expand > scale_table).sum(-1)
        return indexes

    # -------------------------------------------------------------
    # Low-level entropy coder helpers
    # (used later in compress()/decompress() of AHT model)
    # -------------------------------------------------------------
    @staticmethod
    def compress_symbols(symbols, indexes, quantized_cdf, cdf_length, offset, encoder):
        """
        symbols, indexes: int tensors (B,C,H,W)
        encoder: rANS encoder from repo.
        """
        encoder.encode_with_indexes(
            symbols.reshape(-1).int().cpu().numpy(),
            indexes.reshape(-1).int().cpu().numpy(),
            quantized_cdf,
            cdf_length,
            offset,
        )
        return True

    @staticmethod
    def decompress_symbols(indexes, quantized_cdf, cdf_length, offset, decoder):
        """
        Inverse of compress_symbols.
        """
        values = decoder.decode_stream(
            indexes.reshape(-1).int().cpu().numpy(),
            quantized_cdf,
            cdf_length,
            offset,
        )
        outputs = torch.tensor(
            values, device=indexes.device, dtype=torch.float32
        ).reshape(indexes.size())
        return outputs

    # -------------------------------------------------------------
    # Quantization helpers
    # -------------------------------------------------------------
    @staticmethod
    def ste_round(x: torch.Tensor) -> torch.Tensor:
        """
        Straight-through estimator for rounding.
        """
        xr = torch.round(x)
        return (xr - x).detach() + x

    @staticmethod
    def add_noise(x: torch.Tensor) -> torch.Tensor:
        """
        Uniform noise in [-0.5, 0.5] for training-time quantization.
        """
        half = float(0.5)
        noise = torch.empty_like(x).uniform_(-half, half)
        return x + noise

    def quantize_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        Quantize hyperlatents z (factorized).
        - During training: add uniform noise.
        - During eval/inference: round.
        """
        if self.training:
            return self.add_noise(z)
        return torch.round(z)

    def quantize_residual(self, y: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Quantize residual y - mu, then add mu back.
        This matches the GGM assumption around mean μ.
        """
        y_res = y - mu
        if self.training:
            y_q = self.ste_round(y_res)
        else:
            y_q = torch.round(y_res)
        return y_q + mu

    # -------------------------------------------------------------
    # State dict loading
    # -------------------------------------------------------------
    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Just use the nn.Module implementation; keep as hook if you
        need custom buffer update policy later.
        """
        super().load_state_dict(state_dict, strict)
