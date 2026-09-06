import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import torch_dct as dct
from torch.nn.parallel import DistributedDataParallel

from src.models.base_aht import BB as basemodel
from src.layers import PConvRB, conv2x2_down, deconv2x2_up, conv4x4_down, deconv4x4_up, conv3x3_same, deconv3x3_same
from src.layers.conv import *
from src.utils.dct import dctLayer, idctLayer


indices_nga_4x4 = [
                5, 21, 20, 17, 1, 4, 16, 0,
                25, 9, 22, 6, 24, 8, 2, 18,
                26, 10, 29, 28, 13, 12, 23, 3,
                19, 7, 30, 27, 14, 11, 31, 15]

energy_props_nga_4x4 = np.array([
                0.05243656, 0.05239884, 0.05199771, 0.05199648, 0.05196691, 0.05195392, 0.05158095, 0.05138119,
                0.04503353, 0.04499872, 0.04497314, 0.04494835, 0.04471355, 0.04467467, 0.04463650, 0.04462454,
                0.03771098, 0.03760796, 0.01270533, 0.01268872, 0.01268246, 0.01266787, 0.01266614, 0.01264959,
                0.01264733, 0.01264714, 0.01057431, 0.01054387, 0.01051681, 0.01048368, 0.00345573, 0.00343664
            ])

indices_sandia_4x4 = [
                15, 31, 11, 27, 14, 30, 10, 26,  
                7, 23, 13, 29,  6,  9, 22, 25,  
                3, 19, 12,  5, 28, 21,  2,  8, 
                18, 24,  4,  1,  0, 20, 17, 16]

energy_props_sandia_4x4 = np.array([
                0.111992955, 0.10822382, 0.07668176, 0.07386412, 0.068610266, 0.065879494, 0.046987437, 0.044794858, 
                0.04019878, 0.038350187, 0.0358051, 0.033997547, 0.025002223, 0.024836678, 0.023395132, 0.023240933, 
                0.016422447, 0.0151698515, 0.014588387, 0.013730173, 0.0134129515, 0.012311397, 0.010686431, 0.010633184, 
                0.009420066, 0.00935294, 0.006717675, 0.0066061723, 0.0054495297, 0.0052662194, 0.0052283565, 0.00314297
            ])

# Pull below array into only a few lines and remove extra spaces
indices_nga_8x8 = [
    32, 96, 34, 98, 33, 97, 36, 100,
    35, 99, 80, 16, 24, 88, 82, 18,
    84, 81, 20, 17, 64, 0, 26, 90,
    72, 8, 25, 89, 92, 83, 19, 28,
    66, 2, 74, 10, 65, 27, 1, 91,
    73, 76, 12, 9, 68, 4, 3, 67,
    11, 75, 40, 104, 42, 106, 41, 105,
    107, 43, 108, 44, 101, 37, 21, 85,
    93, 29, 77, 13, 69, 5, 109, 45,
    48, 112, 114, 50, 113, 49, 115, 51,
    116, 52, 102, 38, 86, 22, 94, 30,
    78, 14, 70, 6, 117, 110, 53, 46,
    118, 54, 120, 56, 121, 122, 58, 57,
    123, 59, 124, 60, 71, 7, 87, 103,
    23, 39, 95, 15, 79, 31, 125, 61,
    111, 47, 126, 62, 55, 119, 63, 127
]

energy_props_nga_8x8 = np.array([
    0.015269597, 0.0152207175, 0.014988906, 0.014956032, 0.014768486, 0.014720343, 0.014494619, 0.01448476,
    0.014419543, 0.014385894, 0.014216976, 0.014191128, 0.013879825, 0.013863815, 0.013755879, 0.013747389,
    0.013686289, 0.01365581, 0.013634179, 0.013602411, 0.013547689, 0.01353126, 0.013497538, 0.013493595,
    0.013435866, 0.013397105, 0.013391161, 0.013387667, 0.013341518, 0.013340646, 0.01332049, 0.013315556,
    0.013175814, 0.013144272, 0.013085293, 0.013068517, 0.01306135, 0.0130011905, 0.012990219, 0.012986085,
    0.012983699, 0.012963784, 0.012940612, 0.012909133, 0.012834311, 0.012808168, 0.012725956, 0.012723264,
    0.012702313, 0.012694056, 0.01258447, 0.012538389, 0.012228686, 0.012192073, 0.012025812, 0.012009598,
    0.011674223, 0.011663036, 0.01143871, 0.011367285, 0.010796331, 0.010754072, 0.010155486, 0.0101362225,
    0.010011049, 0.009990386, 0.009631671, 0.009612051, 0.009496586, 0.009436892, 0.008270029, 0.008222017,
    0.0035961573, 0.0035909354, 0.0034619542, 0.0034562554, 0.0034288182, 0.0034243625, 0.0033090154, 0.0032991732,
    0.0031854275, 0.003136186, 0.0029780902, 0.0029637523, 0.002813867, 0.0028046046, 0.0027968695, 0.0027895279,
    0.0026901513, 0.0026820241, 0.0026687002, 0.0026457296, 0.002310124, 0.0022948945, 0.0022874621, 0.0022765202,
    0.0007835363, 0.0007760133, 0.00045794618, 0.0004486055, 0.0004412402, 0.00044099035, 0.00043890547, 0.00043759285,
    0.00043000837, 0.00042885935, 0.0004213291, 0.00041908023, 0.00040825075, 0.00040696224, 0.00040676026, 0.00040450128,
    0.00040418046, 0.00040210402, 0.00040184063, 0.00040143984, 0.0004007107, 0.0003997955, 0.00038029044, 0.00037908924,
    0.000372349, 0.00037114613, 0.00029366184, 0.0002915809, 0.00028979182, 0.00028762102, 0.0002538335, 0.00025360298,
])

indices_sandia_8x8 = [
    63, 127, 55, 119, 62, 126, 54, 47,
    118, 111, 61, 46, 125, 110, 53, 117,
    39, 103, 45, 109, 38, 60, 102, 124,
    52, 116, 37, 31, 44, 101, 95, 108,
    30, 59, 94, 123, 51, 115, 36, 100,
    29, 43, 93, 23, 107, 87, 22, 58,
    86, 122, 50, 28, 35, 114, 99, 92,
    15, 21, 42, 79, 85, 106, 57, 14,
    121, 49, 78, 113, 27, 34, 20, 41,
    13, 91, 98, 84, 105, 77, 7, 71,
    33, 6, 12, 26, 19, 56, 97, 70,
    76, 48, 90, 120, 83, 5, 112, 40,
    25, 0, 11, 69, 18, 104, 4, 89,
    32, 75, 17, 82, 10, 68, 96, 24,
    3, 9, 8, 81, 16, 74, 2, 1,
    67, 88, 64, 73, 80, 66, 72, 65,
]

energy_props_sandia_8x8 = np.array([
    0.03409219, 0.03306062, 0.031331144, 0.030313237, 0.029041829, 0.028083188, 0.026691632, 0.026008388,
    0.025673665, 0.025099143, 0.022401366, 0.022036217, 0.0215993, 0.021274649, 0.020583339, 0.019759096,
    0.018937605, 0.018297588, 0.016986225, 0.016296301, 0.016109833, 0.01567656, 0.015498113, 0.015005739,
    0.014377227, 0.013741218, 0.012399066, 0.0121383825, 0.0118798865, 0.011861422, 0.011630373, 0.011326225,
    0.010364337, 0.01007797, 0.009885878, 0.009624297, 0.00927143, 0.008802673, 0.008687516, 0.008246823,
    0.008023139, 0.0076879207, 0.007585207, 0.0075158775, 0.0072656465, 0.0071111564, 0.0064477595, 0.006371543,
    0.006054371, 0.006003583, 0.005886923, 0.0056761517, 0.005675523, 0.0055126324, 0.0053020455, 0.0052865916,
    0.0052134455, 0.0050407364, 0.0049238834, 0.0048877522, 0.0046638427, 0.0045741554, 0.004530953, 0.004488836,
    0.004222735, 0.004208332, 0.0041671726, 0.0038880615, 0.003775196, 0.0036981548, 0.0036309445, 0.0035543758,
    0.0035400072, 0.0034204072, 0.0033581683, 0.0032803274, 0.0032339522, 0.0032221035, 0.00300382, 0.0027449983,
    0.0027216126, 0.0026157626, 0.0025994186, 0.002536365, 0.0024885351, 0.002413604, 0.0023935216, 0.002345732,
    0.0022822875, 0.0022692466, 0.0021998899, 0.0021685592, 0.002151352, 0.0021034216, 0.0020089536, 0.001971933,
    0.0019351515, 0.0018821312, 0.0018483693, 0.0018255546, 0.0017620012, 0.0016929065, 0.0016066888, 0.001594125,
    0.0015893504, 0.001520295, 0.001429943, 0.001414055, 0.0013848434, 0.0013129166, 0.0012818548, 0.0012391362,
    0.0012255695, 0.0012092675, 0.001061365, 0.0010579921, 0.0010543806, 0.0010294796, 0.0010119755, 0.0010026332,
    0.000904892, 0.00089600804, 0.00085399824, 0.0008071143, 0.0006538859, 0.0006510246, 0.0005769089, 0.00056381646,
])


class PadLayer(nn.Module):
    def __init__(self, padding, mode='constant', value=0):
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.value = value

    def forward(self, x):
        return F.pad(x, self.padding, mode=self.mode, value=self.value)


class GConv(nn.Module):
    def __init__(self, dataset: str, N=80, G=4):
        super().__init__()
        self.N = N
        self.G = G

        if dataset == "nga":
            indices = indices_nga_4x4
            energy_props = energy_props_nga_4x4

        elif dataset == "sandia":
            indices = indices_sandia_4x4
            energy_props = energy_props_sandia_4x4

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if self.G == 6:
            self.N_p = [33, 29, 6, 8, 3, 1]
            self.grp_sizes = [8, 8, 2, 8, 4, 2]
        else:
            group_size = int(len(indices) / self.G)
            groups_var = [energy_props[i:i+group_size] for i in range(0, len(energy_props), group_size)]
            group_energy_var_props = [group.sum() for group in groups_var]

            self.N_p = [int(round(prop * self.N)) for prop in group_energy_var_props]
            self.N_p = self.adjust_filters()
            self.grp_sizes = [group_size] * self.G

        self.indices = torch.tensor(indices)
        self.convs = nn.ModuleList(
            [conv3x3_same(in_ch, out_ch) for in_ch, out_ch in zip(self.grp_sizes, self.N_p)]
        )
        print(f"Filter allocation: {self.N_p}")

    def forward(self, x):
        idx = (
            self.indices.view(1, -1, 1, 1)
            .expand(x.size(0), -1, x.size(2), x.size(3))
            .to(x.device)
        )
        x_sorted = torch.gather(x, dim=1, index=idx)
        groups = torch.split(x_sorted, self.grp_sizes, dim=1)

        x_out = [conv(g) for conv, g in zip(self.convs, groups)]
        x_out = torch.cat(x_out, dim=1)

        assert x_out.shape[1] == self.N, f"Output channels ({x_out.shape[1]}) must match N ({self.N})"

        return x_out
    
    def adjust_filters(self):
        for i in reversed(range(len(self.N_p))):
            if self.N_p[i] == 0:
                self.N_p[i] += 1
            else:
                break

        tot = sum(self.N_p)
        if tot <= self.N:
            diff = self.N - tot
            self.N_p[0] += diff
        else:
            diff = tot - self.N
            for j in reversed(range(len(self.N_p))):
                if self.N_p[j] > 1:
                    self.N_p[j] -= 1
                    diff -= 1
                    if diff <= 0:
                        break

        assert sum(self.N_p) == self.N, f"Sum of filters ({sum(self.N_p)}) does not equal N={self.N}"
        
        return self.N_p

    

# -------------------------------------------------------------
# Analysis transform g_a  (FastNIC-style, Fig. 2)
# -------------------------------------------------------------
class g_a(nn.Module):
    def __init__(self, dataset: str, M: int = 320, G: int = 4, latent_dct=False):
        super().__init__()

        mlp_ratio = 3
        partial_ratio = 4
        self.latent_dct = latent_dct

        self.branch = nn.Sequential(
            # (B, C, H, W) --> (B, C*b*b, H/b, W/b) = (B, 32, 64, 64)
            dctLayer(block_size=4),

            # (B, C*b*b, H/b, W/b) --> (B, 80, H/b, W/b) = (B, 80, 64, 64)
            GConv(dataset, N=80, G=G),

            # (B, 80, H/b, W/b) --> (B, 160, H/2b, W/2b) = (B, 160, 32, 32)
            conv2x2_down(80, 160),
            PConvRB(160, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(160, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),
            PConvRB(160, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

            # (B, 160, H/2b, W/2b) --> (B, M, H/4b, W/4b) = (B, 320, 16, 16)
            conv2x2_down(160, M),
        )

    def forward(self, x):
        y = self.branch(x)                  # (1, 320, H/16, W/16)

        # ---------------- DCT Transform ----------------
        if self.latent_dct:
            y = y.permute(0, 2, 3, 1)       # (1, H/16, W/16, 320)
            y = dct.dct(y, norm='ortho')    # (1, H/16, W/16, 320)
            y = y.permute(0, 3, 1, 2)       # (1, 320, H/16, W/16)

        return y


# -------------------------------------------------------------
# Synthesis transform g_s  (mirror of g_a, Fig. 2)
# -------------------------------------------------------------
class g_s(nn.Module):
    def __init__(self, M: int = 320, latent_dct=False):
        super().__init__()

        mlp_ratio = 3
        partial_ratio = 4
        self.latent_dct = latent_dct

        self.branch = nn.Sequential(
            # (B, M, H/16, W/16) --> (B, 160, H/8, W/8) = (B, 160, 32, 32)
            deconv2x2_up(M, 160),
            PConvRB(160, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

            # (B, 160, H/8, W/8) --> (B, 80, H/4, W/4) = (B, 80, 64, 64)
            deconv2x2_up(160, 80),
            PConvRB(80, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

            # (B, 80, H/4, W/4) --> (B, 32, H/4, W/4) = (B, 32, 64, 64)
            deconv3x3_same(80, 32),
            PConvRB(32, mlp_ratio=mlp_ratio, partial_ratio=partial_ratio),

            # (B, C*b*b, H/b, W/b) --> (B, C, H, W) = (B, 2, 256, 256)
            idctLayer(block_size=4),
        )

    def forward(self, y_hat):
        # ---------------- DCT Transform ----------------
        if self.latent_dct:
            y_hat = y_hat.permute(0, 2, 3, 1)       # (1, H/16, W/16, 320)
            y_hat = dct.idct(y_hat, norm='ortho')   # (1, H/16, W/16, 320)
            y_hat = y_hat.permute(0, 3, 1, 2)       # (1, 320, H/16, W/16)

        x_hat = self.branch(y_hat)                  # (1, 320, H/16, W/16)

        return x_hat


# -------------------------------------------------------------
# AHT Hyper-Encoder h_a  (Fig. 2 + Eqs. 3–6)
# y -> [y0..y3] -> z  (N=192)
# -------------------------------------------------------------
class h_a(nn.Module):
    def __init__(self, M: int = 320, N: int = 192):
        super().__init__()
        assert M % 4 == 0, "M must be divisible by 4."

        self.M = M
        self.N = N
        self.group_ch = M // 4   # 80

        # Internal width J corresponds to Conv k2s2 64 blocks
        J = 64

        # C0..C3: Conv k2s2 80
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
    def __init__(self, M: int = 320, N: int = 192):
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
        if isinstance(model, DistributedDataParallel):
            model = model.module

        y = model.g_a(x)          # (1, M, H/16, W/16)

        # # Apply DCT here for testing
        # y = y.permute(0, 2, 3, 1)           # (1, H/16, W/16, 320)
        # y_dct = dct.dct(y, norm='ortho')    # (1, H/16, W/16, 320)
        # y_dct = y_dct.permute(0, 3, 1, 2)   # (1, 320, H/16, W/16)

        groups = model.split_groups(y)
        # groups = model.split_groups(y_dct)

        energies = []
        for g in groups:
            e = torch.mean(g ** 2).item()
            energies.append(e)

    return energies


# -------------------------------------------------------------
# FINAL PACT MODEL
# -------------------------------------------------------------
class PACTModel(basemodel):
    def __init__(self, dataset: str, M: int = 320, N: int = 192, G: int = 4, latent_dct=False):
        super().__init__(N)
        
        self.dataset = dataset
        self.M = M
        self.N = N
        self.G = G

        self.g_a = g_a(
            dataset=dataset,
            M=M,
            G=G,
            latent_dct=latent_dct
        )
        self.g_s = g_s(M=M, latent_dct=latent_dct)

        self.h_a = h_a(M=M, N=N)
        self.h_s = h_s(M=M, N=N)

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
        # from src.entropy_models import ubransEncoder
        from compressai.ans import BufferedRansEncoder as ubransEncoder

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
        # from src.entropy_models import ubransDecoder
        from compressai.ans import RansDecoder as ubransDecoder

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
