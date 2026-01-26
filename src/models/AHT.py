import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_aht import BB as basemodel
from src.layers import PConvRB, conv2x2_down, deconv2x2_up, conv4x4_down, deconv4x4_up, conv3x3_same, deconv3x3_same
from src.utils.dct import dctLayer, idctLayer


class PadLayer(nn.Module):
    def __init__(self, padding, mode='constant', value=0):
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.value = value

    def forward(self, x):
        return F.pad(x, self.padding, mode=self.mode, value=self.value)
    
# -------------------------------------------------------------
# Analysis transform g_a  (FastNIC-style, Fig. 2)
# -------------------------------------------------------------
class g_a(nn.Module):
    def __init__(self, M: int = 256, dct: bool = False):
        super().__init__()

        mlp_ratio = 3
        partial_ratio = 4

        if dct:
            # Changed first two conv2x2_down to conv3x3_same (k3s1p1)
            self.branch = nn.Sequential(
                # (B, C, H, W) --> (B, C*b*b, H/b, W/b) = (B, 32, 64, 64)
                dctLayer(block_size=4),

                # (B, C*b*b, H/b, W/b) --> (B, 32, H/b, W/b) = (B, 32, 64, 64)
                conv3x3_same(2*4*4, 32),
                PConvRB(32, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 32, H/b, W/b) --> (B, 64, H/b, W/b) = (B, 64, 64, 64)
                conv3x3_same(32, 64),
                PConvRB(64, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 64, H/b, W/b) --> (B, 128, H/2b, W/2b) = (B, 128, 32, 32)
                conv2x2_down(64, 128),
                PConvRB(128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
                PConvRB(128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
                PConvRB(128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 128, H/2b, W/2b) --> (B, M, H/4b, W/4b) = (B, 256, 16, 16)
                conv2x2_down(128, M),
            )
        else:
            self.branch = nn.Sequential(
                # (B, C, H, W) --> (B, 32, H/2, W/2) = (B, 32, 128, 128)
                conv2x2_down(2, 32),
                PConvRB(32, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 32, H/2, W/2) --> (B, 64, H/4, W/4) = (B, 64, 64, 64)
                conv2x2_down(32, 64),
                PConvRB(64, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 64, H/4, W/4) --> (B, 128, H/8, W/8) = (B, 128, 32, 32)
                conv2x2_down(64, 128),
                PConvRB(128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
                PConvRB(128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
                PConvRB(128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 128, H/8, W/8) --> (B, M, H/16, W/16) = (B, 256, 16, 16)
                conv2x2_down(128, M),
            )

    def forward(self, x):
        return self.branch(x)


# -------------------------------------------------------------
# Synthesis transform g_s  (mirror of g_a, Fig. 2)
# -------------------------------------------------------------
class g_s(nn.Module):
    def __init__(self, M: int = 256, dct: bool = False):
        super().__init__()

        mlp_ratio = 3
        partial_ratio = 4

        if dct:
            # Replaced last two deconv2x2_up with deconv3x3_same
            self.branch = nn.Sequential(
                # (B, M, H/16, W/16) --> (B, 128, H/8, W/8) = (B, 128, 32, 32)
                deconv2x2_up(M, 128),
                PConvRB(128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 128, H/8, W/8) --> (B, 64, H/4, W/4) = (B, 64, 64, 64)
                deconv2x2_up(128, 64),
                PConvRB(64, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 64, H/4, W/4) --> (B, 32, H/4, W/4) = (B, 32, 64, 64)
                deconv3x3_same(64, 32),
                PConvRB(32, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 32, H/4, W/4) --> (B, C*b*b, H/4, W/4) = (B, 32, 64, 64)
                deconv3x3_same(32, 2*4*4),

                # (B, C*b*b, H/b, W/b) --> (B, C, H, W) = (B, 2, 256, 256)
                idctLayer(block_size=4),
            )
        else:
            self.branch = nn.Sequential(
                # (B, M, H/16, W/16) --> (B, 128, H/8, W/8) = (B, 128, 32, 32)
                deconv2x2_up(M, 128),
                PConvRB(128, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 128, H/8, W/8) --> (B, 64, H/4, W/4) = (B, 64, 64, 64)
                deconv2x2_up(128, 64),
                PConvRB(64, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 64, H/4, W/4) --> (B, 32, H/2, W/2) = (B, 32, 128, 128)
                deconv2x2_up(64, 32),
                PConvRB(32, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

                # (B, 32, H/2, W/2) --> (B, 2, H, W) = (B, 2, 256, 256)
                deconv2x2_up(32, 2),
            )

    def forward(self, y_hat):
        return self.branch(y_hat)


# -------------------------------------------------------------
# AHT Hyper-Encoder h_a  (Fig. 2 + Eqs. 3–6)
# y -> [y0..y3] -> z  (N=192)
# -------------------------------------------------------------
class h_a(nn.Module):
    def __init__(self, M: int = 256, N: int = 192):
        super().__init__()
        assert M % 4 == 0, "M must be divisible by 4."

        self.M = M
        self.N = N
        self.group_ch = M // 4   # 64

        # Internal width J corresponds to Conv k2s2 64 blocks
        J = 64

        # C0..C3: Conv k2s2 64
        self.c0 = conv2x2_down(self.group_ch, J)
        self.c1 = conv2x2_down(self.group_ch, J)
        self.c2 = conv2x2_down(self.group_ch, J)
        self.c3 = conv2x2_down(self.group_ch, J)

        mlp_ratio = 3
        partial_ratio = 4

        # P0 on 64 ch, P1 on 128, P2 on 192
        self.p0 = PConvRB(J,        mlp_ratio=mlp_ratio, partial_ratio=partial_ratio)      # 64
        self.p1 = PConvRB(2 * J,    mlp_ratio=mlp_ratio, partial_ratio=partial_ratio)      # 128
        self.p2 = PConvRB(3 * J,    mlp_ratio=mlp_ratio, partial_ratio=partial_ratio)      # 192

        # C4: Conv k2s2 192 (input 4*J = 256 -> N = 192)
        self.c4 = conv2x2_down(4 * J, N)

    def forward(self, y):
        B, C, H, W = y.shape
        assert C == self.M

        g = self.group_ch
        y0 = y[:, 0:g, :, :]
        y1 = y[:, g:2 * g, :, :]
        y2 = y[:, 2 * g:3 * g, :, :]
        y3 = y[:, 3 * g:4 * g, :, :]

        # z0 = P0(C0(y0))
        z0 = self.p0(self.c0(y0))

        # z1 = P1(Cat(z0, C1(y1)))
        z1_in = torch.cat([z0, self.c1(y1)], dim=1)    # 64 + 64 = 128
        z1 = self.p1(z1_in)

        # z2 = P2(Cat(z1, C2(y2)))
        z2_in = torch.cat([z1, self.c2(y2)], dim=1)    # 128 + 64 = 192
        z2 = self.p2(z2_in)

        # z = C4(Cat(z2, C3(y3)))
        z3_in = torch.cat([z2, self.c3(y3)], dim=1)    # 192 + 64 = 256
        z = self.c4(z3_in)                             # -> (B, 192, H/64, W/64)

        return z


# -------------------------------------------------------------
# AHT Hyper-Decoder h_s  (Fig. 2 + Eqs. 7–10)
# z_hat (B,192,H/64,W/64) -> (mu, alpha) (B,256,H/16,W/16)
# -------------------------------------------------------------
class h_s(nn.Module):
    def __init__(self, M: int = 256, N: int = 192):
        super().__init__()
        assert M % 4 == 0, "M must be divisible by 4."

        self.M = M
        self.N = N
        self.group_ch = M // 4     # 64

        hidden = 256               # matches TConv k2s2 256 in Fig. 2

        mlp_ratio = 3
        partial_ratio = 4

        # Trunk: T4 (192 -> 256, H/64 -> H/32)
        self.t4 = deconv2x2_up(N, hidden)

        # Three PConvRBs along the trunk (P'2, P'1, P'0)
        self.p2 = PConvRB(hidden, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio)
        self.p1 = PConvRB(hidden, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio)
        self.p0 = PConvRB(hidden, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio)

        # T0..T3: each TConv k2s2 128 in the paper
        # Here we output 2 * group_ch = 128 channels so that:
        # (mu_i, alpha_i) = split along channel dim
        out_ch = 2 * self.group_ch  # 128

        self.t3 = deconv2x2_up(hidden, out_ch)   # shallowest (uses only T4)
        self.t2 = deconv2x2_up(hidden, out_ch)   # passes P'2
        self.t1 = deconv2x2_up(hidden, out_ch)   # passes P'2 + P'1
        self.t0 = deconv2x2_up(hidden, out_ch)   # passes P'2 + P'1 + P'0

    def forward(self, z_hat):
        # Base feature after T4: (B,256,H/32,W/32)
        f3 = self.t4(z_hat)

        # Group 3 (lowest energy, shallowest path): uses T4 only
        out3 = self.t3(f3)

        # Group 2: extra PConvRB (P'2)
        f2 = self.p2(f3)
        out2 = self.t2(f2)

        # Group 1: P'2 + P'1
        f1 = self.p1(f2)
        out1 = self.t1(f1)

        # Group 0 (highest energy, deepest): P'2 + P'1 + P'0
        f0 = self.p0(f1)
        out0 = self.t0(f0)

        # Each out_i: (B, 2*group_ch, H/16, W/16)
        mu0, alpha0 = torch.chunk(out0, 2, dim=1)
        mu1, alpha1 = torch.chunk(out1, 2, dim=1)
        mu2, alpha2 = torch.chunk(out2, 2, dim=1)
        mu3, alpha3 = torch.chunk(out3, 2, dim=1)

        # Concatenate in the order [y0,y1,y2,y3] to align with y-channel grouping
        mu     = torch.cat([mu0, mu1, mu2, mu3], dim=1)
        scales = torch.cat([alpha0, alpha1, alpha2, alpha3], dim=1)

        return mu, scales



def compute_group_energy(model, x):
    with torch.no_grad():
        y = model.g_a(x)          # (1, M, H/16, W/16)

        groups = model.split_groups(y)

        energies = []
        for g in groups:
            e = torch.mean(g ** 2).item()
            energies.append(e)

    return energies


# -------------------------------------------------------------
# FINAL AHT MODEL
# -------------------------------------------------------------
class AHTModel(basemodel):
    def __init__(self, M: int = 256, N: int = 192, dct: bool = False):
        super().__init__(N)
        
        self.M = M
        self.N = N
        self.dct = dct

        self.g_a = g_a(M, dct=self.dct)
        self.g_s = g_s(M, dct=self.dct)

        self.h_a = h_a(M, N)
        self.h_s = h_s(M, N)

    def split_groups(self, tensor):
        B, C, H, W = tensor.shape
        g = C // 4
        return [
            tensor[:, 0:g],
            tensor[:, g:2*g],
            tensor[:, 2*g:3*g],
            tensor[:, 3*g:4*g],
        ]

    def forward(self, x, size_check=False):
        # ---------------- Main analysis ----------------
        y = self.g_a(x)

        # ---------------- Hyper encoder ----------------
        z = self.h_a(y)

        # Quantize z around learned global (means_hyper, scales_hyper)
        if self.training:
            z_res = z - self.means_hyper
            z_hat = self.ste_round(z_res) + self.means_hyper
            z_likelihoods = self.entropy_estimation(
                self.add_noise(z_res), self.scales_hyper
            )
        else:
            z_res_hat = torch.round(z - self.means_hyper)
            z_hat = z_res_hat + self.means_hyper
            z_likelihoods = self.entropy_estimation(
                z_res_hat, self.scales_hyper
            )

        # ---------------- Hyper decoder (AHT) ----------------
        mu, scales = self.h_s(z_hat)  # per-channel μ, α for y

        # ---------------- Quantize y around μ ----------------
        if self.training:
            y_res = y - mu
            y_hat = self.ste_round(y_res) + mu
            y_likelihoods = self.entropy_estimation(
                self.add_noise(y_res), scales
            )
        else:
            y_res_hat = torch.round(y - mu)
            y_hat = y_res_hat + mu
            y_likelihoods = self.entropy_estimation(
                y_res_hat, scales
            )

        # ---------------- Reconstruction ----------------
        x_hat = self.g_s(y_hat)

        groups_y  = self.split_groups(y)
        groups_mu = self.split_groups(mu)

        ea_groups = []
        for yi, mui in zip(groups_y, groups_mu):
            ea_groups.append(torch.mean(torch.abs(yi - mui)))

        if size_check:
            print(f"-- x: {list(x.size())}")
            print(f"-- y: {list(y.size())}")
            print(f"-- z: {list(z.size())}")
            print(f"-- z_hat: {list(z_hat.size())}")
            print(f"-- z_likelihoods: {list(z_likelihoods.size())}")
            print(f"-- mu: {list(mu.size())}")
            print(f"-- scales: {list(scales.size())}")
            print(f"-- y_hat: {list(y_hat.size())}")
            print(f"-- y_likelihoods: {list(y_likelihoods.size())}")
            print(f"-- x_hat: {list(x_hat.size())}")

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "y": y,
            "mu": mu,
            "scales": scales,
            "ea_groups": ea_groups,
        }


    # -------------------------------------------------------------
    # Bitstream I/O: compress / decompress
    # -------------------------------------------------------------
    def compress(self, x):
        """
        Compress a single image tensor x in [0,1], shape (1,2,H,W).
        Returns:
            {
              "strings": [y_string, z_string],
              "shape": (H_z, W_z)  # spatial size of z
            }
        """
        from src.entropy_models import ubransEncoder

        # Make sure CDF tables are ready
        if self.quantized_cdf_y.numel() == 0 or self.quantized_cdf_z.numel() == 0:
            raise RuntimeError(
                "CDF tables are empty. Call `model.update(scale_table)` before compress()."
            )

        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            # --------- Analysis & hyper analysis ----------
            y = self.g_a(x)          # (1, M, H/16, W/16)
            z = self.h_a(y)          # (1, N, H/64, W/64)

            # --------- Hyperlatents z: factorized GGM ----------
            z_res_hat = torch.round(z - self.means_hyper)  # integer residuals
            indexes_z = self.build_indexes_z(z_res_hat.size()).to(device)

            encoder_z = ubransEncoder()
            self.compress_symbols(
                z_res_hat,
                indexes_z,
                self.quantized_cdf_z.cpu().numpy(),
                self.cdf_length_z.cpu().numpy(),
                self.offset_z.cpu().numpy(),
                encoder_z,
            )
            z_string = encoder_z.flush()

            z_hat = z_res_hat + self.means_hyper

            # --------- Hyper decoder: get mu, scales for y ----------
            mu, scales = self.h_s(z_hat)   # both (1, M, H/16, W/16)

            # --------- Main latents y: residual around mu ----------
            y_res_hat = torch.round(y - mu)  # integer residuals

            # Map predicted scales -> indices into scale_table
            indexes_y = self.build_indexes_conditional(scales).to(device)

            encoder_y = ubransEncoder()
            self.compress_symbols(
                y_res_hat,
                indexes_y,
                self.quantized_cdf_y.cpu().numpy(),
                self.cdf_length_y.cpu().numpy(),
                self.offset_y.cpu().numpy(),
                encoder_y,
            )
            y_string = encoder_y.flush()

        # Only spatial size of z is needed to reconstruct shapes
        z_shape_hw = z_res_hat.size()[2:]
        return {"strings": [y_string, z_string], "shape": z_shape_hw}

    def decompress(self, strings, shape):
        """
        Decompress bitstreams back to an image.

        Args:
            strings: [y_string, z_string]
            shape: (H_z, W_z) spatial size of z (same as compress output["shape"])

        Returns:
            {"x_hat": x_hat}  with x_hat in [0,1], shape (1,3,H,W)
        """
        from src.entropy_models import ubransDecoder

        self.eval()
        device = self.quantized_cdf_z.device

        if self.quantized_cdf_y.numel() == 0 or self.quantized_cdf_z.numel() == 0:
            raise RuntimeError(
                "CDF tables are empty. Call `model.update(scale_table)` before decompress()."
            )

        with torch.no_grad():
            # --------- Decode hyperlatents z ----------
            # We only support batch size 1 here (same as test.py)
            H_z, W_z = shape
            C_z = self.scales_hyper.size(1)
            output_size = (1, C_z, H_z, W_z)

            indexes_z = self.build_indexes_z(output_size).to(device)

            decoder_z = ubransDecoder()
            decoder_z.set_stream(strings[1])
            z_res_hat = self.decompress_symbols(
                indexes_z,
                self.quantized_cdf_z.cpu().numpy(),
                self.cdf_length_z.cpu().numpy(),
                self.offset_z.cpu().numpy(),
                decoder_z,
            ).to(device)

            z_hat = z_res_hat + self.means_hyper

            # --------- Hyper decoder: get mu, scales for y ----------
            mu, scales = self.h_s(z_hat)  # (1, M, H/16, W/16)

            # --------- Decode main latents y ----------
            indexes_y = self.build_indexes_conditional(scales).to(device)

            decoder_y = ubransDecoder()
            decoder_y.set_stream(strings[0])
            y_res_hat = self.decompress_symbols(
                indexes_y,
                self.quantized_cdf_y.cpu().numpy(),
                self.cdf_length_y.cpu().numpy(),
                self.offset_y.cpu().numpy(),
                decoder_y,
            ).to(device)

            y_hat = y_res_hat + mu

            # --------- Synthesis transform ----------
            x_hat = self.g_s(y_hat).clamp_(0.0, 1.0)

        return {"x_hat": x_hat}
