import os
import sys
import math
import glob
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from typing import Dict, Any
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from pytorch_msssim import ms_ssim

from thop import profile
from src.models.AHT import AHTModel
from src.models.AHT import compute_group_energy


       
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
    
def load_image(filepath: str, min_val: float = -5000.0, max_val: float = 5000.0):
    # W x H x C
    img_np = np.load(filepath).astype(np.float32)
    # C x W x H
    img_np = np.stack([img_np[:,:,0], img_np[:,:,1]], axis=0)
    
    img = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
    img = (img - min_val) / (max_val - min_val)
    return img

def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255):
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, max_val: int = 255):
    metrics: Dict[str, Any] = {}
    org = (org * max_val).clamp(0, max_val).round()
    rec = (rec * max_val).clamp(0, max_val).round()
    metrics["psnr"] = psnr(org, rec).item()
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


# -------------------------------------------------------------
#  CALCULATING kMACs
# -------------------------------------------------------------
def report_total_kmacs():
    model = AHTModel().eval()

    # same training patch
    x = torch.randn(1, 2, 256, 256)

    macs, params = profile(model, inputs=(x,), verbose=False)

    H, W = x.shape[2], x.shape[3]
    kmacs_per_pixel = macs / (H * W * 1000.0)

    print("Total Params:", params)
    print("Total MACs:", macs)
    print("kMACs/pixel:", kmacs_per_pixel)


class DecoderOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.h_s = model.h_s
        self.g_s = model.g_s

    def forward(self, inputs):
        y, z = inputs
        mu, scales = self.h_s(z)
        return self.g_s(y + mu)


def report_decoder_kmacs():
    M, N = 256, 192
    H, W = 256, 256

    model = AHTModel(M=M, N=N).eval()
    dec   = DecoderOnly(model).eval()

    # Latent shapes: y (M, H/16, W/16), z (N, H/64, W/64)
    y = torch.randn(1, M, H // 16, W // 16)
    z = torch.randn(1, N, H // 64, W // 64)

    macs, params = profile(dec, inputs=((y, z),), verbose=False)
    kmacs = macs / (H * W * 1000.0)

    print("Params: ",params)
    print("Decoder MACs:", macs)
    print("Decoder kMACs/pixel:", kmacs)

def report_component_kmacs():
    M,N = 256,192
    H,W = 256,256
    model = AHTModel(M=M,N=N).eval()

    x = torch.randn(1, 2, H, W)
    y = torch.randn(1, M, H//16, W//16)
    z = torch.randn(1, N, H//64, W//64)

    macs_ga, _ = profile(model.g_a, inputs=(x,), verbose=False)
    print("g_a kMAC/pixel:", macs_ga / (H * W * 1000))

    # g_s alone
    macs_gs, _ = profile(model.g_s, inputs=(y,), verbose=False)
    print("g_s kMAC/pixel:", macs_gs/(H*W*1000))

    macs_ha, _ = profile(model.h_a, inputs=(y,), verbose=False)
    print("h_a kMAC/pixel:", macs_ha / (H * W * 1000))

    # h_s alone
    macs_hs, _ = profile(model.h_s, inputs=(z,), verbose=False)
    print("h_s kMAC/pixel:", macs_hs/(H*W*1000))




# -------------------------------------------------------------
# TEST starts here
# -------------------------------------------------------------
def test(args):
    device = torch.device("cuda")
    
    images_list = os.listdir(os.path.abspath(args.dataset))
    assert len(images_list) > 0, f"No files found in {args.dataset}"
    images_list = [os.path.join(args.dataset, f) for f in images_list if f.endswith('.npy')]

    ##### load model
    import importlib
    net = importlib.import_module(f'.{args.model_name}', f'src.models').AHTModel
        
    args.checkpoint = [args.checkpoint]
    # suggest:
    # args.checkpoint = [
    #     '/path-to-ckpt/0.0018.pth.tar', 
    #     '/path-to-ckpt/0.0035.pth.tar', 
    #     '/path-to-ckpt/0.0067.pth.tar', 
    #     '/path-to-ckpt/0.013.pth.tar', 
    #     '/path-to-ckpt/0.025.pth.tar', 
    #     '/path-to-ckpt/0.0483.pth.tar', 
    # ]
    bpp_all = []
    psnr_all = []
    ssim_all = []
    for ckpt in args.checkpoint:
        print("Loading", ckpt)
        checkpoint = torch.load(ckpt, map_location=device)
        model = net()
        model.eval()
        model.load_state_dict(checkpoint, strict=True)
        model.update(get_scale_table(0.12, 64, args.num))
        model = model.to(device)

        bpp_loss = AverageMeter()
        psnr = AverageMeter()
        ssim = AverageMeter()
        y_bpp = AverageMeter()
        z_bpp = AverageMeter()
        enc_time = AverageMeter()
        dec_time = AverageMeter()

        energy_1 = AverageMeter()
        energy_2 = AverageMeter()
        energy_3 = AverageMeter()
        energy_4 = AverageMeter()

        for img_path in tqdm(sorted(images_list)):
            x = load_image(img_path)
            h, w = x.shape[2], x.shape[3]
            x = x.to(device)
            # p = 256
            # x_pad = pad(x, p)
            img_name = img_path.split('/')[-1]
            # print(img_name)
            torch.cuda.synchronize()
            enc_start = time.time()
            with torch.no_grad():
                energies = compute_group_energy(model, x) # REMOVED PAD
                # print("group energies:", energies)
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
            
            
            # # Save reconstruction
            # save_rec = (x_hat.clamp(0,1) * 255).round().byte().cpu().squeeze(0).permute(1,2,0)
            # Image.fromarray(save_rec.numpy()).save(f"{args.dataset}output/recon_{img_name}")

            psnr_img = compute_metrics(x, x_hat, 255)['psnr']

            msssim = ms_ssim(x_hat, x, data_range=1.0).item()

            # msssim = ms_ssim(x_hat, x, data_range=1.0)
            # msssim_db = 10 * (torch.log(1 * 1 / (1 - msssim)) / np.log(10)).item()

            num_pixels = h*w
            bpp_img = sum(len(s) for s in out_enc["strings"]) * 8.0 / num_pixels
            ybpp_img = len(out_enc["strings"][0]) * 8.0 / num_pixels
            zbpp_img = len(out_enc["strings"][1]) * 8.0 / num_pixels

            # print('image:',img_name)
            # print(
            #     f"{img_name}"
            #     f"\tPSNR: {psnr_img} |"
            #     f"\tMS-SSIM: {msssim} |"
            #     f"\tBpp loss: {bpp_img} |"
            #     f"\ty bpp: {ybpp_img} |"
            #     f"\tz bpp: {zbpp_img} |"
            #     f"\tenc time: {enc_t} |"
            #     f"\tdec time: {dec_t} |"
            # )

            bpp_loss.update(bpp_img)
            psnr.update(psnr_img)
            ssim.update(msssim)
            y_bpp.update(ybpp_img)
            z_bpp.update(zbpp_img)
            enc_time.update(enc_t)
            dec_time.update(dec_t)
            energy_1.update(energies[0])
            energy_2.update(energies[1])
            energy_3.update(energies[2])
            energy_4.update(energies[3])

        print(
            f"Test:"
            f"\n--PSNR: {psnr.avg}"
            f"\n--MS-SSIM: {ssim.avg}"
            f"\n--Bpp loss: {bpp_loss.avg}"
            f"\n--y bpp: {y_bpp.avg}"
            f"\n--z bpp: {z_bpp.avg}"
            f"\n--enc time: {enc_time.avg}"
            f"\n--dec time: {dec_time.avg}"
            f"\n--Energy (Grp 1): {energy_1.avg}"
            f"\n--Energy (Grp 2): {energy_2.avg}"
            f"\n--Energy (Grp 3): {energy_3.avg}"
            f"\n--Energy (Grp 4): {energy_4.avg}"
        )
        bpp_all.append(bpp_loss.avg)
        psnr_all.append(psnr.avg)
        ssim_all.append(ssim.avg)
    # print(bpp_all)
    # print(psnr_all)
    # print(ssim_all)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.013, help="Bit-rate distortion parameter (default: %(default)s)")
    parser.add_argument("--model_name", type=str, default="AHT")
    parser.add_argument("--checkpoint", type=str, default="epoch_best.pth.tar", help="Path to a checkpoint")
    parser.add_argument("-num", "--num", type=int, default=60)
    parser.add_argument("-data", "--dataset", type=str, default="/scratch/zb7df/data/NGA/multi_pol/validation")
    args = parser.parse_args()
    # print(args)

    # Reporting KMAX
    report_component_kmacs()
    report_decoder_kmacs()
    report_total_kmacs()

    pol = "HH"
    args.dataset = f"{args.dataset}/gt_{pol}"
    args.checkpoint = f"/scratch/zb7df/checkpoints/AHT_DCT/AHT_lmbda{args.lmbda}/{args.checkpoint}"

    test(args)
