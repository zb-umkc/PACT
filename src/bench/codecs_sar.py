# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import platform
import subprocess
import sys
import time
import math
from tempfile import mkstemp
import numpy as np
import torch
from compressai.utils.bench.codecs import Codec, VTM, HM, AV1


def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size


def iq_norm(I_chan, Q_chan, min_val=-5000, max_val=5000):
    I_norm = (I_chan - min_val) / (max_val - min_val)
    Q_norm = (Q_chan - min_val) / (max_val - min_val)
    return I_norm, Q_norm

def iq_to_ap(I_chan, Q_chan):
    amp_max_val = math.sqrt((5000 ** 2) + ((-5000) ** 2))
    amp  = np.sqrt(I_chan**2 + Q_chan**2) / amp_max_val
    phase = np.arctan2(Q_chan, I_chan)
    return amp, phase


def mse_psnr(target, pred, max_val=1):
    mse  = float(np.mean((target - pred) ** 2))
    psnr = 10 * np.log10(max_val**2 / (mse + 1e-10))
    return mse, psnr


def sqnr(target, pred, neighborhood_size=5):
    device = torch.device("cpu")
    target = torch.from_numpy(target).to(device).unsqueeze(0)
    pred = torch.from_numpy(pred).to(device).unsqueeze(0)
    signal_power = torch.nn.functional.conv2d((target**2), 
                                              torch.ones(1, 1, neighborhood_size, neighborhood_size))
    noise = target - pred
    noise_power = torch.nn.functional.conv2d((noise**2), 
                                             torch.ones(1, 1, neighborhood_size, neighborhood_size))
    sqnr = torch.mean(10*torch.log10((signal_power+1e-10)/(noise_power+1e-10)))
    
    return sqnr.item()


def mae(target, pred):
    device = torch.device("cpu")
    target = torch.from_numpy(target).to(device)
    pred = torch.from_numpy(pred).to(device)
    mae = torch.mean(torch.abs(target - pred))
    return mae.item()


def run_command(cmd, ignore_returncodes=None):
    cmd = [str(c) for c in cmd]
    try:
        rv = subprocess.check_output(cmd)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)


def _get_ffmpeg_version():
    rv = run_command(["ffmpeg", "-version"])
    return rv.split()[2]


class SARBinaryCodec(Codec):
    """SAR-adapted BinaryCodec: encodes I and Q channels separately from .npy."""

    BITDEPTH = 16
    MAX_VAL = (1 << BITDEPTH) - 1  # 65535 — JPEG2000 supports 16-bit natively

    fmt = None

    @classmethod
    def setup_args(cls, _parser):
        pass

    def run(self, in_filepath, quality, metrics=None, return_rec=False):
        info, rec = self._run_impl(in_filepath, quality)
        if return_rec:
            return info, rec
        return info

    def _run_impl(self, in_filepath, quality):
        data = np.load(in_filepath)
        I_chan = data[:, :, 0].astype(np.float32)
        Q_chan = data[:, :, 1].astype(np.float32)
        H, W = I_chan.shape

        total_bits = 0.0
        enc_time = 0.0
        dec_time = 0.0
        recons = []

        for chan in (I_chan, Q_chan):
            cmin, cmax = float(chan.min()), float(chan.max())
            drange = cmax - cmin if cmax > cmin else 1.0
            arr = np.round(
                (chan - cmin) / drange * self.MAX_VAL
            ).clip(0, self.MAX_VAL).astype(np.uint16)

            # Write raw uint16 grayscale
            fd_in, raw_in = mkstemp(suffix=".raw")
            fd_out, raw_out = mkstemp(suffix=".raw")
            _, compressed = mkstemp(suffix=self.fmt)

            with os.fdopen(fd_in, "wb") as f:
                f.write(arr.tobytes())

            start = time.time()
            run_command(self._get_encode_cmd(raw_in, W, H, quality, compressed))
            enc_time += time.time() - start

            total_bits += filesize(compressed) * 8.0

            start = time.time()
            run_command(self._get_decode_cmd(compressed, raw_out, W, H))
            dec_time += time.time() - start

            recon_int = np.fromfile(raw_out, dtype=np.uint16).reshape(H, W)
            recon_float = recon_int.astype(np.float32) / self.MAX_VAL * drange + cmin
            recons.append(recon_float)

            os.close(fd_out)
            for path in (raw_in, raw_out, compressed):
                os.unlink(path)

        I_orig_norm, Q_orig_norm = iq_norm(I_chan, Q_chan)
        I_recon_norm, Q_recon_norm = iq_norm(recons[0], recons[1])
        amp_orig, phase_orig = iq_to_ap(I_chan, Q_chan)
        amp_recon, phase_recon = iq_to_ap(recons[0], recons[1])

        mse_iq, psnr_iq = mse_psnr(
            np.stack((I_orig_norm, Q_orig_norm), axis=2), 
            np.stack((I_recon_norm, Q_recon_norm), axis=2)
        )
        mse_amp, psnr_amp = mse_psnr(amp_orig, amp_recon)
        sqnr_amp = sqnr(amp_orig, amp_recon)
        mae_phase = mae(phase_orig, phase_recon)
        mse_nrcs, _ = mse_psnr(amp_orig**2, amp_recon**2)

        denom = H * W * 2
        bpp = total_bits / denom # Bits per pixel per band

        out = {
            "bpp": round(bpp, 4),
            "encoding_time": round(enc_time, 6),
            "decoding_time": round(dec_time, 6),
            "mse_iq": round(mse_iq, 4),
            "psnr_iq": round(psnr_iq, 4),
            "mse_amp": round(mse_amp, 4),
            "psnr_amp": round(psnr_amp, 4),
            "sqnr_amp": round(sqnr_amp, 4),
            "mae_phase": round(mae_phase, 4),
            "mse_nrcs": round(mse_nrcs, 6),
        }

        return out, tuple(recons)

    def _get_encode_cmd(self, in_filepath, W, H, quality, out_filepath):
        raise NotImplementedError()

    def _get_decode_cmd(self, in_filepath, out_filepath, W, H):
        raise NotImplementedError()


class SARJPEG2000(SARBinaryCodec):

    fmt = ".jp2"

    @property
    def name(self):
        return "JPEG2000"

    @property
    def description(self):
        return f"JPEG2000. ffmpeg version {_get_ffmpeg_version()}"

    def _get_encode_cmd(self, in_filepath, W, H, quality, out_filepath):
        return [
            "ffmpeg", "-loglevel", "panic", "-y",
            "-f", "rawvideo",
            "-pixel_format", "gray16le",
            "-video_size", f"{W}x{H}",
            "-i", in_filepath,
            "-vcodec", "jpeg2000",
            "-pix_fmt", "gray16le",
            "-c:v", "libopenjpeg",
            "-compression_level", quality,
            out_filepath,
        ]

    def _get_decode_cmd(self, in_filepath, out_filepath, W, H):
        return [
            "ffmpeg", "-loglevel", "panic", "-y",
            "-i", in_filepath,
            "-f", "rawvideo",
            "-pixel_format", "gray16le",
            out_filepath,
        ]


def get_vtm_encoder_path(build_dir):
    system = platform.system()
    try:
        elfnames = {"Darwin": "EncoderApp", "Linux": "EncoderAppStatic"}
        return os.path.join(build_dir, elfnames[system])
    except KeyError as err:
        raise RuntimeError(f'Unsupported platform "{system}"') from err


def get_vtm_decoder_path(build_dir):
    system = platform.system()
    try:
        elfnames = {"Darwin": "DecoderApp", "Linux": "DecoderAppStatic"}
        return os.path.join(build_dir, elfnames[system])
    except KeyError as err:
        raise RuntimeError(f'Unsupported platform "{system}"') from err


class SARVTM(VTM):

    BITDEPTH = 12
    MAX_VAL = (1 << BITDEPTH) - 1  # 4095

    def run(self, in_filepath, quality, metrics=None, return_rec=False):
        # Override to skip the PIL-based compute_metrics call in base class
        info, rec = self._run_impl(in_filepath, quality)
        if return_rec:
            return info, rec
        return info

    def _run_impl(self, in_filepath, quality):
        # Load I and Q channels from .npy instead of PIL image
        data = np.load(in_filepath)
        I_chan = data[:, :, 0].astype(np.float32)
        Q_chan = data[:, :, 1].astype(np.float32)

        total_bits = 0.0
        recons = []
        enc_time = 0.0
        dec_time = 0.0

        for chan in (I_chan, Q_chan):
            H, W = chan.shape

            # Normalize to [0, 4095] instead of [0, 255]
            cmin, cmax = float(chan.min()), float(chan.max())
            drange = cmax - cmin if cmax > cmin else 1.0
            arr = np.round(
                (chan - cmin) / drange * self.MAX_VAL
            ).clip(0, self.MAX_VAL).astype(np.uint16)  # uint16 instead of uint8

            fd, yuv_path = mkstemp(suffix=".yuv")
            out_filepath = os.path.splitext(yuv_path)[0] + ".bin"
            fd2, recon_path = mkstemp(suffix=".yuv")

            with os.fdopen(fd, "wb") as f:
                f.write(arr.tobytes())

            # Encode — chroma 400 instead of 444, 12-bit throughout
            cmd = [
                self.encoder_path,
                "-i", yuv_path,
                "-c", self.config_path,
                "-q", quality,
                "-o", "/dev/null",
                "-b", out_filepath,
                "-wdt", W,
                "-hgt", H,
                "-fr", "1",
                "-f", "1",
                "--InputChromaFormat=400",
                "--InputBitDepth=12",
                "--InternalBitDepth=12",
                "--OutputBitDepth=12",
            ]

            print(" ".join([str(x) for x in cmd]))

            start = time.time()
            run_command(cmd)
            enc_time += time.time() - start

            total_bits += filesize(out_filepath) * 8.0
            os.unlink(yuv_path)

            # Decode
            cmd = [self.decoder_path, "-b", out_filepath, "-o", recon_path, "--OutputBitDepth=12"]
            start = time.time()
            run_command(cmd)
            dec_time += time.time() - start

            # Read reconstructed — uint16 instead of uint8
            recon_int = np.fromfile(recon_path, dtype=np.uint16).reshape(H, W)
            recon_float = recon_int.astype(np.float32) / self.MAX_VAL * drange + cmin

            recons.append(recon_float)
            os.close(fd2)
            os.unlink(recon_path)
            os.unlink(out_filepath)

        I_orig_norm, Q_orig_norm = iq_norm(I_chan, Q_chan)
        I_recon_norm, Q_recon_norm = iq_norm(recons[0], recons[1])
        amp_orig, phase_orig = iq_to_ap(I_chan, Q_chan)
        amp_recon, phase_recon = iq_to_ap(recons[0], recons[1])

        mse_iq, psnr_iq = mse_psnr(
            np.stack((I_orig_norm, Q_orig_norm), axis=2), 
            np.stack((I_recon_norm, Q_recon_norm), axis=2)
        )
        mse_amp, psnr_amp = mse_psnr(amp_orig, amp_recon)
        sqnr_amp = sqnr(amp_orig, amp_recon)
        mae_phase = mae(phase_orig, phase_recon)
        mse_nrcs, _ = mse_psnr(amp_orig**2, amp_recon**2)

        denom = H * W * 2
        bpp = total_bits / denom # Bits per pixel per band

        out = {
            "bpp": round(bpp, 4),
            "encoding_time": round(enc_time, 6),
            "decoding_time": round(dec_time, 6),
            "mse_iq": round(mse_iq, 4),
            "psnr_iq": round(psnr_iq, 4),
            "mse_amp": round(mse_amp, 4),
            "psnr_amp": round(psnr_amp, 4),
            "sqnr_amp": round(sqnr_amp, 4),
            "mae_phase": round(mae_phase, 4),
            "mse_nrcs": round(mse_nrcs, 6),
        }

        return out, tuple(recons)


class SARHM(HM):

    BITDEPTH = 10
    MAX_VAL = (1 << BITDEPTH) - 1  # 1023

    def run(self, in_filepath, quality, metrics=None, return_rec=False):
        # Override to skip the PIL-based compute_metrics call in base class
        info, rec = self._run_impl(in_filepath, quality)
        if return_rec:
            return info, rec
        return info

    def _run_impl(self, in_filepath, quality):
        # Load I and Q channels from .npy instead of PIL image
        data = np.load(in_filepath)
        I_chan = data[:, :, 0].astype(np.float32)
        Q_chan = data[:, :, 1].astype(np.float32)

        total_bits = 0.0
        recons = []
        enc_time = 0.0
        dec_time = 0.0

        for chan in (I_chan, Q_chan):
            H, W = chan.shape

            # Normalize to [0, 4095] instead of [0, 255]
            cmin, cmax = float(chan.min()), float(chan.max())
            drange = cmax - cmin if cmax > cmin else 1.0
            arr = np.round(
                (chan - cmin) / drange * self.MAX_VAL
            ).clip(0, self.MAX_VAL).astype(np.uint16)  # uint16 instead of uint8

            fd, yuv_path = mkstemp(suffix=".yuv")
            out_filepath = os.path.splitext(yuv_path)[0] + ".bin"
            fd2, recon_path = mkstemp(suffix=".yuv")

            with os.fdopen(fd, "wb") as f:
                f.write(arr.tobytes())

            # Encode — chroma 400 instead of 444, 12-bit throughout
            cmd = [
                self.encoder_path,
                "-i", yuv_path,
                "-c", self.config_path,
                "-q", quality,
                "-o", "/dev/null",
                "-b", out_filepath,
                "-wdt", W,
                "-hgt", H,
                "-fr", "1",
                "-f", "1",
                "--InputChromaFormat=400",
                f"--InputBitDepth={self.BITDEPTH}",
                f"--InternalBitDepth={self.BITDEPTH}",
                f"--OutputBitDepth={self.BITDEPTH}",
                "--Profile=main-RExt",
            ]

            start = time.time()
            run_command(cmd)
            enc_time += time.time() - start

            total_bits += filesize(out_filepath) * 8.0
            os.unlink(yuv_path)

            # Decode
            cmd = [
                self.decoder_path, 
                "-b", out_filepath, 
                "-o", recon_path, 
                "-d", self.BITDEPTH
            ]
            start = time.time()
            run_command(cmd)
            dec_time += time.time() - start

            # Read reconstructed — uint16 instead of uint8
            recon_int = np.fromfile(recon_path, dtype=np.uint16).reshape(H, W)
            recon_float = recon_int.astype(np.float32) / self.MAX_VAL * drange + cmin

            recons.append(recon_float)
            os.close(fd2)
            os.unlink(recon_path)
            os.unlink(out_filepath)

        I_orig_norm, Q_orig_norm = iq_norm(I_chan, Q_chan)
        I_recon_norm, Q_recon_norm = iq_norm(recons[0], recons[1])
        amp_orig, phase_orig = iq_to_ap(I_chan, Q_chan)
        amp_recon, phase_recon = iq_to_ap(recons[0], recons[1])

        mse_iq, psnr_iq = mse_psnr(
            np.stack((I_orig_norm, Q_orig_norm), axis=2), 
            np.stack((I_recon_norm, Q_recon_norm), axis=2)
        )
        mse_amp, psnr_amp = mse_psnr(amp_orig, amp_recon)
        sqnr_amp = sqnr(amp_orig, amp_recon)
        mae_phase = mae(phase_orig, phase_recon)
        mse_nrcs, _ = mse_psnr(amp_orig**2, amp_recon**2)

        denom = H * W * 2
        bpp = total_bits / denom # Bits per pixel per band

        out = {
            "bpp": round(bpp, 4),
            "encoding_time": round(enc_time, 6),
            "decoding_time": round(dec_time, 6),
            "mse_iq": round(mse_iq, 4),
            "psnr_iq": round(psnr_iq, 4),
            "mse_amp": round(mse_amp, 4),
            "psnr_amp": round(psnr_amp, 4),
            "sqnr_amp": round(sqnr_amp, 4),
            "mae_phase": round(mae_phase, 4),
            "mse_nrcs": round(mse_nrcs, 6),
        }

        return out, tuple(recons)


class SARAV1(AV1):
    """AV1: AOM reference software"""

    BITDEPTH = 12
    MAX_VAL = (1 << BITDEPTH) - 1  # 4095

    def run(self, in_filepath, quality, metrics=None, return_rec=False):
        # Override to skip the PIL-based compute_metrics call in base class
        info, rec = self._run_impl(in_filepath, quality)
        if return_rec:
            return info, rec
        return info

    def _run_impl(self, in_filepath, quality):
        # Load I and Q channels from .npy instead of PIL image
        data = np.load(in_filepath)
        I_chan = data[:, :, 0].astype(np.float32)
        Q_chan = data[:, :, 1].astype(np.float32)

        total_bits = 0.0
        recons = []
        enc_time = 0.0
        dec_time = 0.0

        for chan in (I_chan, Q_chan):
            H, W = chan.shape

            # Normalize to [0, 4095] instead of [0, 255]
            cmin, cmax = float(chan.min()), float(chan.max())
            drange = cmax - cmin if cmax > cmin else 1.0
            arr = np.round(
                (chan - cmin) / drange * self.MAX_VAL
            ).clip(0, self.MAX_VAL).astype(np.uint16)  # uint16 instead of uint8

            fd, yuv_path = mkstemp(suffix=".yuv")
            out_filepath = os.path.splitext(yuv_path)[0] + ".webm"
            fd2, recon_path = mkstemp(suffix=".yuv")

            with os.fdopen(fd, "wb") as f:
                f.write(arr.tobytes())

            # Encode
            cmd = [
                self.encoder_path,
                "--allintra",
                "--full-still-picture-hdr",
                "-w", W,
                "-h", H,
                "--fps=1/1",
                "--limit=1",
                f"--input-bit-depth={self.BITDEPTH}",
                "--cpu-used=0",
                "--threads=1",
                "--passes=1",
                "--end-usage=q",
                "--cq-level=" + str(quality),
                "--monochrome", # YUV format 400 (grayscale)
                "--skip=0",
                "--tune=psnr",
                "--psnr",
                f"--bit-depth={self.BITDEPTH}",
                "-o", out_filepath,
                yuv_path,
            ]

            start = time.time()
            run_command(cmd)
            enc_time = time.time() - start

            total_bits += filesize(out_filepath) * 8.0

            # cleanup encoder input
            os.close(fd)
            os.unlink(yuv_path)

            # Decode
            cmd = [
                self.decoder_path,
                out_filepath, "-o",
                yuv_path,
                "--rawvideo",
                f"--output-bit-depth={self.BITDEPTH}",
            ]

            start = time.time()
            run_command(cmd)
            dec_time = time.time() - start

            # Read reconstructed — uint16 instead of uint8
            recon_int = np.fromfile(recon_path, dtype=np.uint16).reshape(H, W)
            recon_float = recon_int.astype(np.float32) / self.MAX_VAL * drange + cmin

            recons.append(recon_float)
            os.close(fd2)
            os.unlink(recon_path)
            os.unlink(out_filepath)

        I_orig_norm, Q_orig_norm = iq_norm(I_chan, Q_chan)
        I_recon_norm, Q_recon_norm = iq_norm(recons[0], recons[1])
        amp_orig, phase_orig = iq_to_ap(I_chan, Q_chan)
        amp_recon, phase_recon = iq_to_ap(recons[0], recons[1])

        mse_iq, psnr_iq = mse_psnr(
            np.stack((I_orig_norm, Q_orig_norm), axis=2), 
            np.stack((I_recon_norm, Q_recon_norm), axis=2)
        )
        mse_amp, psnr_amp = mse_psnr(amp_orig, amp_recon)
        sqnr_amp = sqnr(amp_orig, amp_recon)
        mae_phase = mae(phase_orig, phase_recon)
        mse_nrcs, _ = mse_psnr(amp_orig**2, amp_recon**2)

        denom = H * W * 2
        bpp = total_bits / denom # Bits per pixel per band

        out = {
            "bpp": round(bpp, 4),
            "encoding_time": round(enc_time, 6),
            "decoding_time": round(dec_time, 6),
            "mse_iq": round(mse_iq, 4),
            "psnr_iq": round(psnr_iq, 4),
            "mse_amp": round(mse_amp, 4),
            "psnr_amp": round(psnr_amp, 4),
            "sqnr_amp": round(sqnr_amp, 4),
            "mae_phase": round(mae_phase, 4),
            "mse_nrcs": round(mse_nrcs, 6),
        }

        return out, tuple(recons)
