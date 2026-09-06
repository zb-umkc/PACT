import argparse
import math
import random
import sys
import os
import time
import numpy as np
from datetime import date
from tqdm import tqdm
from PIL import Image
from pytorch_msssim import ms_ssim, ssim
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.models.PACT import compute_group_energy

# 1. Tells PyTorch to auto-tune and find the fastest GPU math paths
torch.backends.cudnn.benchmark = True

# 2. Ensures PyTorch uses the highest speed math configurations
torch.set_float32_matmul_precision('high')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split, pol, transform, arch=None):
        if dataset == "nga":
            self.data_dir = f"/scratch/zb7df/data/{dataset}/{split}/gt_{pol}"
            self.min_val = -5000.0
            self.max_val = 5000.0
        elif dataset == "sandia":
            self.data_dir = f"/scratch/zb7df/data/{dataset}/{split}"
            self.min_val = -500.0
            self.max_val = 500.0
        else:
            raise ValueError("Unknown dataset structure. Please check the data_path.")
        
        self.dataset_list = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
        self.transform = transform
        self.arch = arch
        
    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.dataset_list[idx])
        # W x H x C
        img_np = np.load(image_path).astype(np.float32)

        # C x W x H
        img_np = np.stack([img_np[:,:,0], img_np[:,:,1]], axis=0)
        if self.arch == "AHT":
            zeros = np.zeros((1, img_np.shape[1], img_np.shape[2]))
            img_np = np.concatenate([img_np, zeros], axis=0)
        
        img = torch.tensor(img_np, dtype=torch.float32)
        img = (img - self.min_val) / (self.max_val - self.min_val)

        if self.transform:
            img = self.transform(img)
        
        return img
    

class LocalLogMSELoss(nn.Module):
    def __init__(self, kernel_size=5, eps=1e-8):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, pred, target):
        sq_error = (target - pred) ** 2

        local_mse = F.avg_pool2d(
            sq_error,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0   # <-- no padding
        )

        return torch.log(local_mse + self.eps).mean()
    

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, iq_loss="mse", alpha=1.0, gamma=1.0, kernel_size=5, eps=1e-8, min_val=-5000.0, max_val=5000.0):
        super().__init__()
        self.iq_loss = iq_loss
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.lmbda = lmbda
        self.ea_weights = torch.tensor([0.0, 0.1, 0.3, 0.5])
        self.gamma = gamma   # EA strength factor (paper uses gamma≈1)
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.eps = eps
        self.min_val = min_val
        self.max_val = max_val

        print(f"Loss Min/Max: {self.min_val}/{self.max_val}")

    def convert_to_amp(self, output_iq, target_iq, min_val, max_val):
        output_iq = torch.clamp(output_iq, 0, 1)

        # (0, 1) -> (min_val, max_val)
        target_denorm = (target_iq * (max_val - min_val)) + min_val
        output_denorm = (output_iq * (max_val - min_val)) + min_val
        amp_max_val = torch.sqrt(torch.tensor(max_val ** 2 + (min_val) ** 2))

        # I/Q -> Amplitude: (0, 1)
        target_amp = torch.sqrt(torch.sum(target_denorm ** 2, dim=1, keepdim=True))/amp_max_val
        output_amp = torch.sqrt(torch.sum(output_denorm ** 2, dim=1, keepdim=True))/amp_max_val
        output_amp = torch.clamp(output_amp, 0, 1)

        return output_amp, target_amp

    def forward(self, output, target):

        N, C, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        if C > 2:
            target = target[:, :2, :, :]
            output["x_hat"] = output["x_hat"][:, :2, :, :]

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out['y_bpp'] = torch.log(output['likelihoods']['y']).sum() / (-math.log(2) * num_pixels)
        out['z_bpp'] = torch.log(output['likelihoods']['z']).sum() / (-math.log(2) * num_pixels)

        # Calculate the EA Loss
        ea_loss = 0.0
        for w, ea in zip(self.ea_weights.to(target.device), output["ea_groups"]):
            ea_loss += w * ea
            
        out["ea_loss"] = ea_loss
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["psnr"] = 10 * (torch.log(1 * 1 / out["mse_loss"]) / np.log(10))
        out["l1_loss"] = self.l1(output["x_hat"], target)
        # _msssim = ms_ssim(output["x_hat"], target, data_range=1.0).item()
        # out["msssim_loss"] = 1 - _msssim
        _ssim = ssim(output["x_hat"], target, data_range=1.0, nonnegative_ssim=True).item()
        out["ssim_loss"] = 1 - _ssim

        # Calculate the Amp Loss (based on SQNR)
        output_amp, target_amp = self.convert_to_amp(output["x_hat"], target, self.min_val, self.max_val)
        sq_error = (target_amp - output_amp) ** 2
        local_mse = F.avg_pool2d(
            sq_error,
            kernel_size=self.kernel_size,
            stride=1
        )
        out["amp_loss"] = torch.log(local_mse + self.eps).mean()

        if self.iq_loss == "mse":
            out["iq_loss"] = out["mse_loss"]

        elif self.iq_loss == "l1_ssim":
            out["iq_loss"] = (out["l1_loss"] + out["ssim_loss"]) / 2
            
        out["distortion_loss"] = (self.alpha * 255**2 * out["iq_loss"]) + ((1 - self.alpha) * out["amp_loss"])
        out["loss"] = self.lmbda * out["distortion_loss"] + out["bpp_loss"] + self.gamma * ea_loss

        return out
    

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # [FIX] Get Python float to prevent graph leak
        if isinstance(val, torch.Tensor):
            val = val.detach()
            if val.numel() > 1:
                val = val.mean()
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                print("Stopping early due to no improvement in validation loss.")


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, global_step, writer, args):
    model.train()
    device = next(model.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    l1_loss = AverageMeter()
    ssim_loss = AverageMeter()
    ea_loss = AverageMeter()
    amp_loss = AverageMeter()
    iq_loss = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    for i, d in enumerate(train_dataloader):
        if args.size_check:
            print("\nTRAINING")
            print(f"-- Train input: {list(d.size())}")

        global_step+=1
        d = d.to(device)
        optimizer.zero_grad()
        out_net = model(d, args.size_check)

        out_criterion = criterion(out_net, d)
        # out_criterion["loss"].backward()
        out_criterion["loss"].mean().backward()
        if args.clip_max_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            if total_norm.isnan() or total_norm.isinf():
                print("non-finite norm, skip this batch")
                continue
        optimizer.step()

        bpp_loss.update(out_criterion["bpp_loss"])
        loss.update(out_criterion["loss"])
        mse_loss.update(out_criterion["mse_loss"])
        psnr.update(out_criterion["psnr"])
        l1_loss.update(out_criterion["l1_loss"])
        ssim_loss.update(out_criterion["ssim_loss"])
        ea_loss.update(out_criterion["ea_loss"])
        amp_loss.update(out_criterion["amp_loss"])
        iq_loss.update(out_criterion["iq_loss"])
        y_bpp.update(out_criterion["y_bpp"])
        z_bpp.update(out_criterion["z_bpp"])

        if args.size_check:
            break

    if not args.size_check:
        print(
            f"-- Train | "
            f"Loss: {loss.avg:.4f} | "
            f"MSE: {mse_loss.avg:.6f} | "
            f"PSNR: {psnr.avg:.3f} | "
            f"L1 Loss: {l1_loss.avg:.4f} | "
            f"SSIM Loss: {ssim_loss.avg:.4f} | "
            f"BPP: {bpp_loss.avg:.4f} | "
            f"EA Loss: {ea_loss.avg:.4f} | "
            f"Amp Loss: {amp_loss.avg:.4f} | "
            f"IQ Loss: {iq_loss.avg:.4f} | "
        )
        torch.cuda.empty_cache()

        if writer is not None:
            writer.add_scalar("Train/Loss", loss.avg, global_step = epoch)
            writer.add_scalar("Train/MSE Loss", mse_loss.avg, global_step = epoch)
            writer.add_scalar("Train/L1 Loss", l1_loss.avg, global_step = epoch)
            writer.add_scalar("Train/SSIM Loss", ssim_loss.avg, global_step = epoch)
            writer.add_scalar("Train/BPP Loss", bpp_loss.avg, global_step = epoch)
            writer.add_scalar("Train/EA Loss", ea_loss.avg, global_step = epoch)
            writer.add_scalar("Train/Amp Loss", amp_loss.avg, global_step = epoch)
            writer.add_scalar("Train/IQ Loss", iq_loss.avg, global_step = epoch)

    return global_step


def val_epoch(epoch, val_dataloader, model, criterion, writer, args):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    l1_loss = AverageMeter()
    ssim_loss = AverageMeter()
    ea_loss = AverageMeter()
    amp_loss = AverageMeter()
    iq_loss = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()
    energy_1 = AverageMeter()
    energy_2 = AverageMeter()
    energy_3 = AverageMeter()
    energy_4 = AverageMeter()

    with torch.no_grad():
        for d in val_dataloader:
            if args.size_check:
                print("\nVALIDATION")
                print(f"-- Validation input: {list(d.size())}")

            d = d.to(device)
            out_net = model(d, args.size_check)
            out_criterion = criterion(out_net, d)

            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            psnr.update(out_criterion["psnr"])
            l1_loss.update(out_criterion["l1_loss"])
            ssim_loss.update(out_criterion["ssim_loss"])
            ea_loss.update(out_criterion["ea_loss"])
            amp_loss.update(out_criterion["amp_loss"])
            iq_loss.update(out_criterion["iq_loss"])
            y_bpp.update(out_criterion["y_bpp"])
            z_bpp.update(out_criterion["z_bpp"])

            energies = compute_group_energy(model, d)
            energy_1.update(energies[0])
            energy_2.update(energies[1])
            energy_3.update(energies[2])
            energy_4.update(energies[3])

            if args.size_check:
                break

    if not args.size_check:
        print(
            f"-- Test  | "
            f"Loss: {loss.avg:.4f} | "
            f"MSE: {mse_loss.avg:.6f} | "
            f"PSNR: {psnr.avg:.3f} | "
            f"L1 Loss: {l1_loss.avg:.4f} | "
            f"SSIM Loss: {ssim_loss.avg:.4f} | "
            f"BPP: {bpp_loss.avg:.4f} | "
            f"EA Loss: {ea_loss.avg:.4f} | "
            f"Amp Loss: {amp_loss.avg:.4f} | "
            f"IQ Loss: {iq_loss.avg:.4f} | "
        )
        print(f"-- Group Energy: {energy_1.avg:.2f} | {energy_2.avg:.2f} | {energy_3.avg:.2f} | {energy_4.avg:.2f}")
        if writer is not None:
            writer.add_scalar("Test/Loss", loss.avg, global_step = epoch)
            writer.add_scalar("Test/MSE Loss", mse_loss.avg, global_step = epoch)
            writer.add_scalar("Test/L1 Loss", l1_loss.avg, global_step = epoch)
            writer.add_scalar("Test/SSIM Loss", ssim_loss.avg, global_step = epoch)
            writer.add_scalar("Test/BPP Loss", bpp_loss.avg, global_step = epoch)
            writer.add_scalar("Test/EA Loss", ea_loss.avg, global_step = epoch)
            writer.add_scalar("Train/Amp Loss", amp_loss.avg, global_step = epoch)
            writer.add_scalar("Train/IQ Loss", iq_loss.avg, global_step = epoch)

    if dist.is_available() and dist.is_initialized():
        loss_sum = torch.tensor(loss.sum, device=device)
        loss_count = torch.tensor(loss.count, device=device)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        return (loss_sum / loss_count).item()

    return loss.avg


def pad_to_multiple(img, k=64):
    if isinstance(img, torch.Tensor):
        # Tensor shape: (C, H, W)
        _, h, w = img.shape
        new_w = (w + k - 1) // k * k
        new_h = (h + k - 1) // k * k
        pad_w = new_w - w
        pad_h = new_h - h
        return torch.nn.functional.pad(img, 
            (0, pad_w, 0, pad_h), # left, right, top, bottom
            mode="reflect"
        )
    else:
        w, h = img.size
        new_w = (w + k - 1) // k * k
        new_h = (h + k - 1) // k * k
        pad_w = new_w - w
        pad_h = new_h - h
        return transforms.functional.pad(img,
            (0, 0, pad_w, pad_h),  # left, top, right, bottom
            padding_mode="reflect"
        )
    

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--run-name", type=str, help="Experiment name in format [arch]/[config]")
    parser.add_argument("-a", "--architecture", type=str, default="PACT", help="Model architecture (PACT or AHT)")
    parser.add_argument("-d", "--dataset", type=str, default="nga", help="Dataset")
    parser.add_argument("-e", "--epochs", default=1, type=int, help="Number of epochs (default: %(default)s)")
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float, help="Learning rate (default: %(default)s)")
    parser.add_argument("-n", "--num-workers", type=int, default=8, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument("--lambda", dest="lmbda", type=float, help="Bit-rate distortion parameter (default: %(default)s)")
    parser.add_argument("--alpha", dest="alpha", type=float, help="Distortion loss weight parameter (default: %(default)s)")
    parser.add_argument("--gamma", dest="gamma", type=float, help="EA strength factor (default: %(default)s)")
    parser.add_argument("-bs", "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size", type=int, default=1, help="Test batch size (default: %(default)s)")
    parser.add_argument("--cuda", default=True, help="Use cuda")
    parser.add_argument("--seed", type=float, default=314, help="Set random seed for reproducibility")
    parser.add_argument("--clip-max-norm", default=1.0, type=float, help="Gradient clipping max norm (default: %(default)s)")
    parser.add_argument("--checkpoint", type=str, help="Full path to a checkpoint")
    parser.add_argument("--resume-optimizer", action="store_true", help="Load optimizer state from the checkpoint")
    parser.add_argument("--resume-scheduler", action="store_true", help="Load scheduler state from the checkpoint")
    parser.add_argument("--reset-lr", action="store_true", help="Reset optimizer learning rate to --learning-rate even when resuming from checkpoint")
    parser.add_argument("--size-check", action="store_true", help="Print tensor sizes instead of training")
    parser.add_argument("--iq-loss", type=str, default="l1_ssim", help="Distortion loss for I/Q component: mse or l1_ssim (default: %(default)s)")
    parser.add_argument("-g", "--groups", type=int, default=8, help="Number of groups for GConv in g_a (default: %(default)s)")
    parser.add_argument("--latent-dct", action="store_true", help="Apply DCT across latent channels")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print(args)

    model_dir = os.path.join("/scratch/zb7df/models/", args.run_name, args.dataset)
    log_dir = os.path.join("/scratch/zb7df/logs/", args.run_name, args.dataset, f"lambda_{args.lmbda}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    if args.size_check:
        print("---------------------")
        print("-- SIZE CHECK MODE --")
        print("---------------------")
        args.epochs = 1
    else:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

    pol = "HH"

    train_dataset = Dataset(
        dataset=args.dataset,
        split="train",
        pol=pol,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]),
        arch=args.architecture,
    )
    val_dataset = Dataset(
        dataset=args.dataset,
        split="validation",
        pol=pol,
        transform=None,
        arch=args.architecture,
    )

    # MULTI-GPU SUPPORT #
    use_ddp = torch.cuda.device_count() > 1 and "LOCAL_RANK" in os.environ
    if use_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=(device == "cuda"),
            drop_last=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=(device == "cuda"),
        )

    if args.architecture == "PACT":
        net = importlib.import_module(".PACT", f'src.models').PACTModel(
            dataset=args.dataset,
            G=args.groups,
            latent_dct=args.latent_dct
        )
    else:
        net = importlib.import_module(".AHT", f'src.models').AHTModel()

    if args.size_check: print(net)
    net = net.to(device)
    if use_ddp:
        net = DistributedDataParallel(
            net,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    last_epoch = 0

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # if args.cuda and torch.cuda.device_count() > 1:
    #     print("--------")
    #     print("WARNING: Multiple GPUs detected - do you want to use CustomDataParallel?")
    #     print("--------")
    #     net = CustomDataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = RateDistortionLoss(
        lmbda=args.lmbda,
        iq_loss=args.iq_loss,
        alpha=args.alpha,
        gamma=args.gamma,
        min_val=train_dataset.min_val,
        max_val=train_dataset.max_val
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        threshold=1e-3
    )
    early_stopping = EarlyStopping(patience=20, delta=1e-3)

    best_loss = float("inf")
    global_step = 0
    last_periodic_ckpt = None
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint        

        if any(k.startswith("module.") for k in state_dict):
            print("-- Checkpoint was trained with DDP")
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            print("-- Currently training with DDP")
            net.module.load_state_dict(state_dict)
        else:
            print("-- Currently training without DDP")
            net.load_state_dict(state_dict)

        if args.resume_optimizer and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(f"Loading optimizer state from checkpoint: LR={optimizer.param_groups[0]['lr']}")
            if args.reset_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = args.learning_rate

        if args.resume_scheduler and isinstance(checkpoint, dict) and "scheduler" in checkpoint:
            print("Loading scheduler state from checkpoint")
            scheduler.load_state_dict(checkpoint["scheduler"])

        if args.reset_lr and not args.resume_optimizer:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.learning_rate

        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            last_epoch = checkpoint["epoch"] + 1
        if isinstance(checkpoint, dict) and "best_loss" in checkpoint:
            best_loss = checkpoint["best_loss"]
        if isinstance(checkpoint, dict) and "global_step" in checkpoint:
            global_step = checkpoint["global_step"]

    if args.size_check:
        writer = None
    elif use_ddp and dist.get_rank() != 0:
        writer = None
    else:
        writer = SummaryWriter(log_dir)

    for epoch in range(last_epoch, (last_epoch + args.epochs)):
        start_time = time.time()
        if not args.size_check:
            print(f"\nStarting epoch {epoch}")
            print(f"-- LR: {optimizer.param_groups[0]['lr']}")

        if use_ddp:
            train_dataloader.sampler.set_epoch(epoch)
        
        global_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            global_step,
            writer,
            args,
        )

        loss = val_epoch(
            epoch,
            val_dataloader,
            net,
            criterion,
            writer,
            args,
        )

        if not args.size_check:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if use_ddp:
                dist.barrier()

            if is_best and (not use_ddp or dist.get_rank() == 0):
                checkpoint = {
                    "epoch": epoch,
                    "model": net.module.state_dict() if use_ddp else net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "global_step": global_step,
                }
                torch.save(checkpoint, os.path.join(model_dir, f"lambda_{args.lmbda}.pth.tar"))

            if use_ddp:
                dist.barrier()

            epoch_time = time.time() - start_time
            print(f"-- Time: {epoch_time:.1f} seconds")

            scheduler.step(loss)
            early_stopping.check_early_stop(loss)
            if early_stopping.stop_training:
                break

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
