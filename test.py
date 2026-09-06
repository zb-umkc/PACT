import os
import sys
import math
import glob
import time
import torch
import argparse
import csv
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Dict, Any
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from pytorch_msssim import ms_ssim
import matplotlib.pyplot as plt
from datetime import date
import importlib
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
from thop import profile
from ptflops import get_model_complexity_info
import sarpy.io.general.nitf as nitf
from skimage.exposure import match_histograms, adjust_gamma

from src.models.PACT import compute_group_energy


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )
    
def load_image(filepath: str, min_val: float = -5000.0, max_val: float = 5000.0, arch=None):
    # W x H x C
    img_np = np.load(filepath).astype(np.float32)

    # C x W x H
    img_np = np.stack([img_np[:,:,0], img_np[:,:,1]], axis=0)
    if arch == "AHT":
        zeros = np.zeros((1, img_np.shape[1], img_np.shape[2]))
        img_np = np.concatenate([img_np, zeros], axis=0)
    
    img = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
    img = (img - min_val) / (max_val - min_val)

    return img

def load_image_nitf(filepath: str, min_val: float = -5000.0, max_val: float = 5000.0):

    sar_data         = nitf.NITFReader(filepath)
    sar_image        = sar_data.read_raw()
    with open(filepath, "rb") as f:
        sar_header   = f.read(sar_data.nitf_details.img_segment_offsets[0])
        
        if 0:
            print("\n****************\n")
            print(sar_header)
            print("\n****************\n")
        f.seek(sar_data.nitf_details.des_subheader_offsets[0])
        sar_metadata = f.read()
        
        if 0:
            print("\n****************\n")
            print(sar_metadata)
            print("\n****************\n")
        
    sar_image        = np.clip(sar_image, min_val, max_val)
    # sar_image        = (sar_image - min_val) / (max_val - min_val)
    sar_image        = torch.tensor(sar_image).permute(2, 0, 1)

    c, h, w = sar_image.shape
    ps = 128

    # Calculate padding required along each dimension
    pad_y = (ps - (h % ps)) % ps
    pad_x = (ps - (w % ps)) % ps

    # Pad the image using reflection (mirror) padding
    padding = (0, pad_x, 0, pad_y)  # (left, right, top, bottom)
    gt_sar = F.pad(sar_image, padding, mode='reflect').permute(1, 2, 0)

    # return {
    #     "img": sar_image,
    #     "sar_header": sar_header,
    #     "sar_metadata": sar_metadata
    # }

    print(f"SHAPE: {gt_sar.shape}, MIN: {gt_sar.min()}, MAX: {gt_sar.max()}")

    # Export as .npy
    np.save(filepath.replace(".nitf", ".npy"), gt_sar.numpy())
    sys.exit()

    return gt_sar.unsqueeze(0)


def sandia2nga_transform(I, Q, nga_reference, use_gamma=False, plot_hist=False):
    """Forward transform: Sandia I/Q -> NGA-domain I/Q"""
    # orig_amp = np.sqrt(I**2 + Q**2).numpy()

    if use_gamma:
        # adjust_gamma expects [0, 1] input
        amp_norm = orig_amp / (orig_amp.max() + 1e-8)
        matched_amp_norm = adjust_gamma(amp_norm, gamma=0.5)  # tune this
        matched_amp = matched_amp_norm * nga_reference.max()
    else:
        # matched_flat = match_histograms(orig_amp.flatten(), nga_reference)
        # matched_amp = matched_flat.reshape(orig_amp.shape)
        I_matched_flat = match_histograms(I.flatten(), nga_reference[0])
        matched_I = I_matched_flat.reshape(I.shape)
        Q_matched_flat = match_histograms(Q.flatten(), nga_reference[1])
        matched_Q = Q_matched_flat.reshape(Q.shape)

    if plot_hist:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].hist(orig_amp.flatten(), bins=100, alpha=0.5)
        axes[0].set_title("Original Amplitude")
        axes[1].hist(nga_reference.flatten(), bins=100, alpha=0.5)
        axes[1].set_title("NGA Reference Amplitude")
        axes[2].hist(matched_amp.flatten(), bins=100, alpha=0.5)
        axes[2].set_title("Matched Amplitude")
        plt.tight_layout()
        plt.savefig("plots/amp_hist_matching.png")

    # Preserve phase, apply new amplitude
    # safe_amp = np.where(orig_amp > 1e-8, orig_amp, 1.0)
    # scale = np.where(orig_amp > 1e-8, matched_amp / safe_amp, 0.0)
    safe_I = np.where(I > 1e-8, I, 1.0)
    scale_I = np.where(I > 1e-8, matched_I / safe_I, 0.0)
    safe_Q = np.where(Q > 1e-8, Q, 1.0)
    scale_Q = np.where(Q > 1e-8, matched_Q / safe_Q, 0.0)

    return I * scale_I, Q * scale_Q, I, Q, matched_I, matched_Q


def sandia2nga_inverse(recon_I, recon_Q, orig_amp):
    """Inverse transform: NGA-domain reconstruction -> Sandia domain"""
    recon_amp = np.sqrt(recon_I**2 + recon_Q**2)

    # Invert the amplitude mapping using original and matched as reference
    inverted_amp = match_histograms(recon_amp, orig_amp)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(recon_amp.flatten(), bins=100, alpha=0.5)
    axes[0].set_title("Reconstructed Amplitude")
    axes[1].hist(orig_amp.flatten(), bins=100, alpha=0.5)
    axes[1].set_title("Original Amplitude (Ref)")
    axes[2].hist(inverted_amp.flatten(), bins=100, alpha=0.5)
    axes[2].set_title("Inverted Amplitude (Final)")
    plt.tight_layout()
    plt.savefig("plots/amp_hist_matching_inv.png")

    safe_amp = np.where(recon_amp > 1e-8, recon_amp, 1.0)
    scale = np.where(recon_amp > 1e-8, inverted_amp / safe_amp, 0.0)
    
    return recon_I * scale, recon_Q * scale


# def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255):
#     return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

def psnr(a, b):
    mse_loss = nn.MSELoss()
    mse = mse_loss(a, b).item()
    # psnr = 10 * torch.log10(1 / mse)
    psnr = 10*np.log10(1/mse)

    return mse, psnr

def sqnr(target, pred, neighborhood_size=5):
    device = torch.device("cpu")
    target = target.to(device)
    pred = pred.to(device)
    signal_power = torch.nn.functional.conv2d((target**2), 
                                              torch.ones(1, 1, neighborhood_size, neighborhood_size))
    noise = target - pred
    noise_power = torch.nn.functional.conv2d((noise**2), 
                                             torch.ones(1, 1, neighborhood_size, neighborhood_size))
    sqnr = torch.mean(10*torch.log10((signal_power+1e-10)/(noise_power+1e-10)))
    
    return sqnr

def phase_error(phase1, phase2):
    return torch.mean(torch.abs(phase1 - phase2))

def compute_metrics(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        min_val: float,
        max_val: float,
        arch: str,
    ) -> Dict[str, Any]:

    print(x.device, x_hat.device)

    metrics: Dict[str, Any] = {}

    if arch == "AHT":
        x = x[:, :2, :, :]
        x_hat = x_hat[:, :2, :, :]

    ### I/Q Error
    # (0, 1)
    orig_iq = x
    rec_iq = torch.clamp(x_hat, 0, 1)
    _, metrics["psnr_iq"] = psnr(orig_iq, rec_iq)
    metrics["msssim_iq"] = ms_ssim(rec_iq, orig_iq, data_range=1.0).item()

    ### Amp Error
    # (0, 1) -> (-5000, 5000)
    orig_denorm = (x * (max_val - min_val)) + min_val
    rec_denorm = (x_hat * (max_val - min_val)) + min_val
    amp_max_val = torch.sqrt(torch.tensor(max_val ** 2 + min_val ** 2))

    # I/Q -> Amplitude: (0, 1)
    orig_amp = torch.sqrt(torch.sum(orig_denorm ** 2, dim=1, keepdim=True))/amp_max_val
    rec_amp = torch.sqrt(torch.sum(rec_denorm ** 2, dim=1, keepdim=True))/amp_max_val
    rec_amp = torch.clamp(rec_amp, 0, 1)

    _, metrics["psnr_amp"] = psnr(orig_amp, rec_amp)
    metrics["sqnr_amp"] = sqnr(orig_amp, rec_amp).item()
    metrics["msssim_amp"] = ms_ssim(rec_amp, orig_amp, data_range=1.0).item()

    ### Phase Error
    # I/Q -> Phase: (-pi, pi)
    orig_phase = torch.atan2(orig_denorm[0, 1, :, :], orig_denorm[0, 0, :, :])
    rec_phase = torch.atan2(rec_denorm[0, 1, :, :], rec_denorm[0, 0, :, :])
    metrics["mae_phase"] = phase_error(orig_phase, rec_phase).item()

    ### NRCS Error
    metrics["mse_nrcs"], _ = psnr(orig_amp**2, rec_amp**2)

    return metrics


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        if type(val) == torch.Tensor:
            val = val.detach().cpu()

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def get_scale_table(min, max, levels):
    """Returns table of logarithmically scales."""
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


def load_checkpoint_compatible(model, checkpoint_path, device):
    """Load checkpoint, handling DDP prefix mismatch."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract the model state dict
    state_dict = checkpoint if isinstance(checkpoint, dict) and "model" not in checkpoint else checkpoint.get("model", checkpoint)
    
    # Remove 'module.' prefix if present (DDP wrapping)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v  # strip 'module.'
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    return model


# -------------------------------------------------------------
#  CALCULATING kMACs
# -------------------------------------------------------------
def report_component_profiles(args=None, show_layers=False):
    M,N = 320,192
    H,W = 256,256
    denom = H*W*1000.0

    if args.architecture == "PACT":
        net = importlib.import_module(".PACT", f'src.models').PACTModel
        input_ch = 2
    else:
        net = importlib.import_module(".AHT", f'src.models').AHTModel
        input_ch = 3

    model = net(
        dataset=args.dataset,
        M=M,
        N=N,
        G=args.groups,
        latent_dct=args.latent_dct
    ).eval()

    x = torch.randn(1, input_ch, H, W)
    y = torch.randn(1, M, H//16, W//16)
    z = torch.randn(1, N, H//64, W//64)

    macs_ga, params_ga = profile(model.g_a, inputs=(x,), verbose=False)
    macs_gs, params_gs = profile(model.g_s, inputs=(y,), verbose=False)
    macs_ha, params_ha = profile(model.h_a, inputs=(y,), verbose=False)
    macs_hs, params_hs = profile(model.h_s, inputs=(z,), verbose=False)

    profiles = {
        "g_a": {
            "macs": macs_ga,
            "params": int(params_ga),
        },
        "g_s": {
            "macs": macs_gs,
            "params": int(params_gs),
        },
        "h_a": {
            "macs": macs_ha,
            "params": int(params_ha),
        },
        "h_s": {
            "macs": macs_hs,
            "params": int(params_hs),
        },
        "enc": {
            "macs": macs_ga + macs_ha,
            "params": int(params_ga + params_ha),
        },
        "dec": {
            "macs": macs_gs + macs_hs,
            "params": int(params_gs + params_hs),
        },
        "total": {
            "macs": macs_ga + macs_ha + macs_gs + macs_hs,
            "params": int(params_ga + params_ha + params_gs + params_hs),
        },
        "denom": denom,
    }

    if show_layers:
        _, _ = get_model_complexity_info(
            model, 
            (input_ch, H, W), 
            as_strings=True, 
            print_per_layer_stat=True,
        )
    
    print(
        f"\n--Total Params | kMAC/px: {profiles['total']['params']} | {profiles['total']['macs']/denom}"
        f"\n----Encoder: {profiles['enc']['params']} | {profiles['enc']['macs']/denom}"
        f"\n------g_a: {profiles['g_a']['params']} | {profiles['g_a']['macs']/denom}"
        f"\n------h_a: {profiles['h_a']['params']} | {profiles['h_a']['macs']/denom}"
        f"\n----Decoder: {profiles['dec']['params']} | {profiles['dec']['macs']/denom}"
        f"\n------g_s: {profiles['g_s']['params']} | {profiles['g_s']['macs']/denom}"
        f"\n------h_s: {profiles['h_s']['params']} | {profiles['h_s']['macs']/denom}"
    )

    return profiles


def report_deepspeed_profile(args=None, show_layers=False):
    M,N = 320,192
    H,W = 256,256
    denom = H*W*1000.0

    if args.architecture == "PACT":
        net = importlib.import_module(".PACT", f'src.models').PACTModel
        model = net(
            dataset=args.dataset,
            M=M, N=N,
            G=args.groups,
            latent_dct=args.latent_dct
        ).eval()
        input_ch = 2
    else:
        net = importlib.import_module(".AHT", f'src.models').AHTModel
        model = net(M=M, N=N).eval()
        input_ch = 3

    x_shape = (1, input_ch, H, W)
    y_shape = (1, M, H//16, W//16)
    z_shape = (1, N, H//64, W//64)

    with get_accelerator().device(0):       
        _, macs_ga, params_ga = get_model_profile(model=model.g_a,
                                            input_shape=x_shape,
                                            print_profile=False,
                                            detailed=False,
                                            warm_up=10,
                                            as_string=False)
        
        _, macs_gs, params_gs = get_model_profile(model=model.g_s,
                                            input_shape=y_shape,
                                            print_profile=False,
                                            detailed=False,
                                            warm_up=10,
                                            as_string=False)
        
        _, macs_ha, params_ha = get_model_profile(model=model.h_a,
                                            input_shape=y_shape,
                                            print_profile=False,
                                            detailed=False,
                                            warm_up=10,
                                            as_string=False)
        
        _, macs_hs, params_hs = get_model_profile(model=model.h_s,
                                            input_shape=z_shape,
                                            print_profile=False,
                                            detailed=False,
                                            warm_up=10,
                                            as_string=False)

    profiles = {
        "g_a": {
            "macs": macs_ga,
            "params": int(params_ga),
        },
        "g_s": {
            "macs": macs_gs,
            "params": int(params_gs),
        },
        "h_a": {
            "macs": macs_ha,
            "params": int(params_ha),
        },
        "h_s": {
            "macs": macs_hs,
            "params": int(params_hs),
        },
        "enc": {
            "macs": macs_ga + macs_ha,
            "params": int(params_ga + params_ha),
        },
        "dec": {
            "macs": macs_gs + macs_hs,
            "params": int(params_gs + params_hs),
        },
        "total": {
            "macs": macs_ga + macs_ha + macs_gs + macs_hs,
            "params": int(params_ga + params_ha + params_gs + params_hs),
        },
        "denom": denom,
    }

    if show_layers:
        _, _, _ = get_model_profile(
            model=model,
            input_shape=x_shape,
            print_profile=True,
            detailed=True,
            warm_up=10,
            as_string=False
        )

    print(
        f"\n--Total Params: {profiles['total']['params']} | kMAC/px: {profiles['total']['macs']/denom}"
        f"\n----Encoder: {profiles['enc']['params']} | {profiles['enc']['macs']/denom}"
        f"\n------g_a: {profiles['g_a']['params']} | {profiles['g_a']['macs']/denom}"
        f"\n------h_a: {profiles['h_a']['params']} | {profiles['h_a']['macs']/denom}"
        f"\n----Decoder: {profiles['dec']['params']} | {profiles['dec']['macs']/denom}"
        f"\n------g_s: {profiles['g_s']['params']} | {profiles['g_s']['macs']/denom}"
        f"\n------h_s: {profiles['h_s']['params']} | {profiles['h_s']['macs']/denom}"
    )

    return profiles


# -------------------------------------------------------------
# TEST starts here
# -------------------------------------------------------------
def test(args, profiles):
    device = torch.device("cuda")
    
    images_list = os.listdir(args.data_dir)
    assert len(images_list) > 0, f"No files found in {args.dataset}/{args.split}"
    images_list = [os.path.join(args.data_dir, f) for f in images_list if f.endswith('.npy')]

    if args.adapt:
        nga_reference = np.load("/scratch/zb7df/data/nga/nga_reference.npy")
        if args.architecture == "PACT":
            net = importlib.import_module(".PACT", f'src.models').PACTModel
            model = net(
                dataset="nga",
                G=args.groups,
                latent_dct=args.latent_dct
            )
        else:
            net = importlib.import_module(".AHT", f'src.models').AHTModel
            model = net()

        ckpt_path = os.path.join("/scratch/zb7df/models/", args.run_name, "nga", f"lambda_{args.lmbda}.pth.tar")

    else:
        if args.architecture == "PACT":
            net = importlib.import_module(".PACT", f'src.models').PACTModel
            model = net(
                dataset=args.dataset,
                G=args.groups,
                latent_dct=args.latent_dct
            )
        else:
            net = importlib.import_module(".AHT", f'src.models').AHTModel
            model = net()

        ckpt_path = os.path.join("/scratch/zb7df/models/", args.run_name, args.dataset, f"lambda_{args.lmbda}.pth.tar")
    
    print(f"Instantiated model for dataset {model.dataset}")
    print(f"Loading checkpoint from {ckpt_path}")
    model.eval()
    model = load_checkpoint_compatible(model, ckpt_path, device)
    model.update(get_scale_table(0.12, 64, args.num))
    model = model.to(device)

    bpp_loss = AverageMeter()
    psnr_iq = AverageMeter()
    msssim_iq = AverageMeter()
    psnr_amp = AverageMeter()
    sqnr_amp = AverageMeter()
    msssim_amp = AverageMeter()
    mae_phase = AverageMeter()
    mse_nrcs = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()
    enc_time = AverageMeter()
    dec_time = AverageMeter()

    energy_1 = AverageMeter()
    energy_2 = AverageMeter()
    energy_3 = AverageMeter()
    energy_4 = AverageMeter()

    for img_path in tqdm(sorted(images_list)):
        if img_path.endswith("nitf"):
            x = load_image_nitf(img_path, min_val=args.min_val, max_val=args.max_val)
        else:
            x = load_image(img_path, min_val=args.min_val, max_val=args.max_val, arch=args.architecture)
            print(x.shape)

        if args.adapt:
            I, Q = x[:, 0, :, :].squeeze(), x[:, 1, :, :].squeeze()
            I_t, Q_t, orig_amp, matched_amp = sandia2nga_transform(I, Q, nga_reference)
            x = torch.tensor(np.stack([I_t, Q_t]), dtype=torch.float32).unsqueeze(0)

        c, h, w = x.shape[1], x.shape[2], x.shape[3]
        x = x.to(device)
        # p = 256
        # x_pad = pad(x, p)
        img_name = img_path.split('/')[-1]
        # print(img_name)
        torch.cuda.synchronize()
        enc_start = time.time()
        with torch.no_grad():
            energies = compute_group_energy(model, x) # REMOVED PAD
            # print("---- Group Energy:", energies)
            out_enc = model.compress(x) # REMOVED PAD
        torch.cuda.synchronize()
        enc_t = time.time() - enc_start
        
        torch.cuda.synchronize()
        dec_start = time.time()
        with torch.no_grad():
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        torch.cuda.synchronize()
        dec_t = time.time() - dec_start
        # x_hat = crop(out_dec["x_hat"], (h,w))
        x_hat = out_dec["x_hat"]  # REMOVED CROP

        if args.adapt:
            x_hat = x_hat.cpu().numpy()
            I_hat, Q_hat = sandia2nga_inverse(
                x_hat[:, 0, :, :].squeeze(), x_hat[:, 1, :, :].squeeze(), orig_amp
            )
            x_hat = torch.tensor(np.stack([I_hat, Q_hat]), dtype=torch.float32).unsqueeze(0)
            x_hat = x_hat.to(device)

        # # Save reconstructed I/Q for visualization
        # save_rec = x_hat.clamp(0,1).float().cpu().squeeze(0).permute(1,2,0).numpy()
        # np.save(f"recon/{img_name}", save_rec)

        # metrics = compute_metrics(x, x_hat, mode="iq") # Calculate in I/Q format
        print(f"Min Val: {args.min_val}, Max Val: {args.max_val}")
        print(f"x min: {x.min()}, x max: {x.max()}")
        print(f"x_hat min: {x_hat.min()}, x_hat max: {x_hat.max()}")
        # print(f"-- 1: {x_hat[:, 0, :, :].min()}, {x_hat[:, 0, :, :].max()}")
        # print(f"-- 2: {x_hat[:, 1, :, :].min()}, {x_hat[:, 1, :, :].max()}")
        # print(f"-- 3: {x_hat[:, 2, :, :].min()}, {x_hat[:, 2, :, :].max()}")
        metrics = compute_metrics(x, x_hat, args.min_val, args.max_val, arch=args.architecture)

        # msssim = ms_ssim(x_hat, x, data_range=1.0)
        # msssim_db = 10 * (torch.log(1 * 1 / (1 - msssim)) / np.log(10)).item()

        # Calculate Bits per Pixel per Band
        #TODO: Hard-coding 2 bands for now (SAR I/Q)
        num_pixels = 2*h*w
        bpp_img = sum(len(s) for s in out_enc["strings"]) * 8.0 / num_pixels
        ybpp_img = len(out_enc["strings"][0]) * 8.0 / num_pixels
        zbpp_img = len(out_enc["strings"][1]) * 8.0 / num_pixels

        bpp_loss.update(bpp_img)
        psnr_iq.update(metrics["psnr_iq"])
        msssim_iq.update(metrics["msssim_iq"])
        psnr_amp.update(metrics["psnr_amp"])
        sqnr_amp.update(metrics["sqnr_amp"])
        msssim_amp.update(metrics["msssim_amp"])
        mae_phase.update(metrics["mae_phase"])
        mse_nrcs.update(metrics["mse_nrcs"])
        y_bpp.update(ybpp_img)
        z_bpp.update(zbpp_img)
        enc_time.update(enc_t)
        dec_time.update(dec_t)
        energy_1.update(energies[0])
        energy_2.update(energies[1])
        energy_3.update(energies[2])
        energy_4.update(energies[3])

    arch = args.run_name.split("/")[0]
    model = args.run_name.split("/")[1]
    test_date = date.today().strftime("%Y%m%d")

    # TODO: Clean this up
    if "test/1024" in args.split:
        results_filename = "results_highres.csv"
    elif "full" in args.split:
        results_filename = "results_full.csv"
    else:
        results_filename = "results.csv"

    fieldnames = ["arch", "model", "dataset", "lmbda", "test_date", "bpp", "psnr_iq", "msssim_iq", "psnr_amp", "sqnr_amp", 
                  "msssim_amp", "mae_phase", "mse_nrcs", "enc_time", "dec_time", 
                  "total_kmac_per_px", "enc_kmac_per_px", "dec_kmac_per_px", "ga_kmac_per_px", "ha_kmac_per_px", 
                  "gs_kmac_per_px", "hs_kmac_per_px", "total_params", "energy_1", "energy_2", "energy_3", "energy_4"]
    
    write_data = {"arch": arch, "model": model, "dataset": args.dataset, "lmbda": args.lmbda, "test_date": test_date, 
                  "bpp": bpp_loss.avg, "psnr_iq": psnr_iq.avg, "msssim_iq": msssim_iq.avg, "psnr_amp": psnr_amp.avg, 
                  "sqnr_amp": sqnr_amp.avg, "msssim_amp": msssim_amp.avg, "mae_phase": mae_phase.avg, "mse_nrcs": mse_nrcs.avg,
                  "enc_time": enc_time.avg, "dec_time": dec_time.avg,
                  "total_kmac_per_px": profiles['total']['macs']/profiles["denom"], "enc_kmac_per_px": profiles['enc']['macs']/profiles["denom"], 
                  "dec_kmac_per_px": profiles['dec']['macs']/profiles["denom"], "ga_kmac_per_px": profiles['g_a']['macs']/profiles["denom"], 
                  "ha_kmac_per_px": profiles['h_a']['macs']/profiles["denom"], "gs_kmac_per_px": profiles['g_s']['macs']/profiles["denom"],
                  "hs_kmac_per_px": profiles['h_s']['macs']/profiles["denom"], "total_params": profiles['total']['params'],
                  "energy_1": energy_1.avg, "energy_2": energy_2.avg, "energy_3": energy_3.avg, "energy_4": energy_4.avg}
    
    with open(results_filename, "a") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:
            writer.writeheader()
        writer.writerow(write_data)

    print(
        f"Test:"
        f"\n--BPP: {bpp_loss.avg}"
        f"\n--PSNR (I/Q): {psnr_iq.avg}"
        f"\n--MS-SSIM (I/Q): {msssim_iq.avg}"
        f"\n--PSNR (Amp): {psnr_amp.avg}"
        f"\n--SQNR (Amp): {sqnr_amp.avg}"
        f"\n--MS-SSIM (Amp): {msssim_amp.avg}"
        f"\n--MAE (Phase): {mae_phase.avg}"
        f"\n--MSE (NRCS): {mse_nrcs.avg}"
        f"\n--y bpp: {y_bpp.avg}"
        f"\n--z bpp: {z_bpp.avg}"
        f"\n--enc time: {enc_time.avg}"
        f"\n--dec time: {dec_time.avg}"
        f"\n--Energy (Grp 1): {energy_1.avg}"
        f"\n--Energy (Grp 2): {energy_2.avg}"
        f"\n--Energy (Grp 3): {energy_3.avg}"
        f"\n--Energy (Grp 4): {energy_4.avg}"
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--lambda", dest="lmbda", type=float, help="Bit-rate distortion parameter (default: %(default)s)")
    parser.add_argument("-d", "--dataset", type=str, default="nga", help="Dataset")
    parser.add_argument("--split", type=str, default="validation", help="Data split for inference")
    parser.add_argument("--run-name", type=str, help="Experiment name in format [arch]/[config]")
    parser.add_argument("-num", "--num", type=int, default=60)
    parser.add_argument("-a", "--architecture", type=str, default="PACT", help="Model architecture (PACT or AHT)")
    parser.add_argument("-g", "--groups", type=int, default=8, help="Number of groups for GConv in g_a (default: %(default)s)")
    parser.add_argument("--adapt", action="store_true", help="Adapt the model to the input data")
    parser.add_argument("--latent-dct", action="store_true", help="Apply DCT across latent channels")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print(args)

    pol = "HH"
    if args.dataset == "nga":
        args.data_dir = f"/scratch/zb7df/data/{args.dataset}/{args.split}/1024/gt_{pol}"
        args.min_val = -5000.0
        args.max_val = 5000.0
        print(f"Min Val: {args.min_val}, Max Val: {args.max_val}")
    elif args.dataset == "sandia":
        args.data_dir = f"/scratch/zb7df/data/{args.dataset}/{args.split}"
        args.min_val = -500.0
        args.max_val = 500.0
        print(f"Min Val: {args.min_val}, Max Val: {args.max_val}")
    else:
        raise ValueError("Unknown dataset structure. Please check the data_path.")

    # profiles = report_component_profiles(args=args, show_layers=False)
    profiles = report_deepspeed_profile(args=args, show_layers=False)

    test(args, profiles)


if __name__ == '__main__':
    main(sys.argv[1:])
