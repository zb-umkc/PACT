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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, pol, transform):
        self.data_dir = f"{data_path}/gt_{pol}"
        self.dataset_list = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f))]
        self.transform = transform
        self.min_val = -5000.0
        self.max_val = 5000.0

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_dir, self.dataset_list[idx])
        # W x H x C
        img_np = np.load(image_path).astype(np.float32)
        # C x W x H
        img_np = np.stack([img_np[:,:,0], img_np[:,:,1]], axis=0)
        
        img = torch.tensor(img_np, dtype=torch.float32)
        img = (img - self.min_val) / (self.max_val - self.min_val)

        if self.transform:
            img = self.transform(img)
        
        return img
    

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2, distortion_loss="mse", alpha=0.5):
        super().__init__()
        self.distortion_loss = distortion_loss
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.lmbda = lmbda
        self.ea_weights = torch.tensor([0.0, 0.1, 0.3, 0.5])
        self.gamma = 1.0   # strength factor (paper uses gamma≈1)
        self.alpha = alpha

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

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

        if self.distortion_loss == "mse":
            out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"] + self.gamma * ea_loss

        elif self.distortion_loss == "l1_ssim":
            out["distortion_loss"] = (self.alpha * out["l1_loss"]) + ((1 - self.alpha) * out["ssim_loss"])
            out["loss"] = self.lmbda * 255**2 * out["distortion_loss"] + out["bpp_loss"] + self.gamma * ea_loss

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
            val = val.detach().item()
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
            # f"y_bpp: {y_bpp.avg:.4f} | "
            # f"z_bpp: {z_bpp.avg:.4f}"
        )
        torch.cuda.empty_cache()

        writer.add_scalar("Train/Loss", loss.avg, global_step = epoch)
        writer.add_scalar("Train/MSE Loss", mse_loss.avg, global_step = epoch)
        writer.add_scalar("Train/L1 Loss", l1_loss.avg, global_step = epoch)
        writer.add_scalar("Train/SSIM Loss", ssim_loss.avg, global_step = epoch)
        writer.add_scalar("Train/BPP Loss", bpp_loss.avg, global_step = epoch)
        writer.add_scalar("Train/EA Loss", ea_loss.avg, global_step = epoch)

    return global_step


def test_epoch(epoch, test_dataloader, model, criterion, writer, args):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr = AverageMeter()
    l1_loss = AverageMeter()
    ssim_loss = AverageMeter()
    ea_loss = AverageMeter()
    y_bpp = AverageMeter()
    z_bpp = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            if args.size_check:
                print("\nTESTING")
                print(f"-- Test input: {list(d.size())}")

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
            y_bpp.update(out_criterion["y_bpp"])
            z_bpp.update(out_criterion["z_bpp"])

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
            # f"y_bpp: {y_bpp.avg:.4f} | "
            # f"z_bpp: {z_bpp.avg:.4f}"
        )
        writer.add_scalar("Test/Loss", loss.avg, global_step = epoch)
        writer.add_scalar("Test/MSE Loss", mse_loss.avg, global_step = epoch)
        writer.add_scalar("Test/L1 Loss", l1_loss.avg, global_step = epoch)
        writer.add_scalar("Test/SSIM Loss", ssim_loss.avg, global_step = epoch)
        writer.add_scalar("Test/BPP Loss", bpp_loss.avg, global_step = epoch)
        writer.add_scalar("Test/EA Loss", ea_loss.avg, global_step = epoch)

    return loss.avg

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("--model_name", type=str, default="AHT")
    parser.add_argument("--model_class", type=str, default="hypers")
    parser.add_argument("-tr_d", "--train_dataset", type=str, default="/scratch/zb7df/data/NGA/multi_pol/train", help="Training dataset")
    parser.add_argument("-te_d", "--test_dataset", type=str, default="/scratch/zb7df/data/NGA/multi_pol/train_val", help="Testing dataset")
    parser.add_argument( "-e", "--epochs", default=2, type=int, help="Number of epochs (default: %(default)s)")
    parser.add_argument( "-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)")
    parser.add_argument( "-n", "--num-workers", type=int, default=8, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument( "--lambda", dest="lmbda", type=float, default=0.013, help="Bit-rate distortion parameter (default: %(default)s)")
    parser.add_argument( "--alpha", dest="alpha", type=float, default=0.5, help="L1-SSIM weight parameter (default: %(default)s)")
    parser.add_argument( "-bs", "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)")
    parser.add_argument( "--test-batch-size", type=int, default=1, help="Test batch size (default: %(default)s)")
    parser.add_argument( "--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate (default: %(default)s)")
    parser.add_argument( "--patch-size", type=int, nargs=2, default=(256, 256), help="Size of the patches to be cropped (default: %(default)s)")
    parser.add_argument("--cuda", default=True, help="Use cuda")
    parser.add_argument( "--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--save_path", type=str, default="/scratch/zb7df/checkpoints/AHT_DCT/", help="Where to Save model")
    parser.add_argument("--log_dir", type=str, default="/scratch/zb7df/checkpoints/AHT_DCT/", help="Where to Save logs")
    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument( "--clip_max_norm", default=1.0, type=float, help="gradient clipping max norm (default: %(default)s")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument( "--size_check", action="store_true", help="Print tensor sizes instead of training")
    parser.add_argument( "--dct", action="store_true", help="Apply DCT transform to images")
    parser.add_argument("--dist", type=str, default="l1_ssim", help="Distortion loss function: mse or l1_ssim (default: %(default)s)")
    args = parser.parse_args(argv)
    return args


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


def main(argv):
    args = parse_args(argv)
    print(args)
    # today = date.today().strftime("%Y%m%d")
    run_name = f"{args.model_name}_lmbda{str(args.lmbda)}"
    args.log_dir = os.path.join(args.log_dir, run_name)
    args.save_path = os.path.join(args.save_path, run_name)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    
    if args.size_check:
        print("---------------------")
        print("-- SIZE CHECK MODE --")
        print("---------------------")
        args.epochs = 1
    else:
        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
        if not os.path.exists(args.save_path): os.makedirs(args.save_path)
        

    pol = "HH"

    train_dataset = Dataset(
        args.train_dataset,
        pol,
        transform=transforms.Compose([
            # transforms.Pad(128, padding_mode="reflect"),
            # transforms.RandomCrop(args.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]),
    )
    test_dataset = Dataset(
        args.test_dataset,
        pol,
        transform=None,
        # transform=transforms.Compose([
        #     lambda img: pad_to_multiple(img, 64),
        # ])
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    import importlib
    net = importlib.import_module(f'.AHT', f'src.models').AHTModel(dct=args.dct)
    if args.size_check: print(net)
    net = net.to(device)

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    last_epoch = 0

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if args.cuda and torch.cuda.device_count() > 1:
        print("--------")
        print("WARNING: Multiple GPUs detected - do you want to use CustomDataParallel?")
        print("--------")
        # net = CustomDataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = RateDistortionLoss(lmbda=args.lmbda, distortion_loss=args.dist, alpha=args.alpha)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        threshold=1e-3
    )
    early_stopping = EarlyStopping(patience=20, delta=1e-3)

    if args.size_check:
        writer = None
    else:
        writer = SummaryWriter(args.log_dir)

    best_loss = float("inf")
    global_step = 0
    for epoch in range(last_epoch, args.epochs):
        start_time = time.time()
        if not args.size_check:
            print(f"\nStarting epoch {epoch}")
            print(f"-- LR: {optimizer.param_groups[0]['lr']}")
        
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

        loss = test_epoch(
            epoch,
            test_dataloader,
            net,
            criterion,
            writer,
            args,
        )

        if not args.size_check:
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if is_best:
                torch.save(net.state_dict(), os.path.join(args.save_path, 'epoch_' +'best' + '.pth.tar'))

            if epoch % 1000 == 0:
                torch.save(net.state_dict(), os.path.join(args.save_path, 'epoch_' + str(epoch) + '.pth.tar'))

            epoch_time = time.time() - start_time
            print(f"-- Time: {epoch_time:.1f} seconds")

            scheduler.step(loss)
            early_stopping.check_early_stop(loss)
            if early_stopping.stop_training:
                break


if __name__ == "__main__":
    main(sys.argv[1:])
