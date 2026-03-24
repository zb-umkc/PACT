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

import abc
import io
import os
import platform
import subprocess
import sys
import time

from tempfile import mkstemp
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import PIL.Image as Image
import torch

from pytorch_msssim import ms_ssim

from compressai.transforms.functional import rgb2ycbcr, ycbcr2rgb
from compressai.utils.bench.codecs import BinaryCodec, JPEG2000, VTM, HM, AV1

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def filesize(filepath: str) -> int:
    """Return file size in bits of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size


def read_image(filepath: str, mode: str = "RGB") -> np.array:
    """Return PIL image in the specified `mode` format."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return Image.open(filepath).convert(mode)


def _compute_psnr(a, b, max_val: float = 255.0) -> float:
    mse = torch.mean((a - b) ** 2).item()
    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return psnr


def _compute_ms_ssim(a, b, max_val: float = 255.0) -> float:
    return ms_ssim(a, b, data_range=max_val).item()


_metric_functions = {
    "psnr-rgb": _compute_psnr,
    "ms-ssim-rgb": _compute_ms_ssim,
}


def compute_metrics(
    a: Union[np.array, Image.Image],
    b: Union[np.array, Image.Image],
    metrics: Optional[List[str]] = None,
    max_val: float = 255.0,
) -> Dict[str, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`."""

    if metrics is None:
        metrics = ["psnr-rgb"]

    def _convert(x):
        if isinstance(x, Image.Image):
            x = np.asarray(x)
        x = torch.from_numpy(x.copy()).float().unsqueeze(0)
        if x.size(3) == 3:
            # (1, H, W, 3) -> (1, 3, H, W)
            x = x.permute(0, 3, 1, 2)
        return x

    a = _convert(a)
    b = _convert(b)

    out = {}
    for metric_name in metrics:
        out[metric_name] = _metric_functions[metric_name](a, b, max_val)
    return out


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


def _get_bpg_version(encoder_path):
    rv = run_command([encoder_path, "-h"], ignore_returncodes=[1])
    return rv.split()[4]


class Codec(abc.ABC):
    """Abstract base class"""

    _description = None

    def __init__(self, args):
        self._set_args(args)

    def _set_args(self, args):
        return args

    @classmethod
    @abc.abstractmethod
    def setup_args(cls, _parser):
        pass

    @property
    def description(self):
        return self._description

    @property
    @abc.abstractmethod
    def name(self):
        raise NotImplementedError()

    def _load_img(self, img):
        return read_image(os.path.abspath(img))

    @abc.abstractmethod
    def _run_impl(self, img, quality, *args, **kwargs):
        raise NotImplementedError()

    def run(
        self,
        in_filepath,
        quality: int,
        metrics: Optional[List[str]] = None,
        return_rec: bool = False,
    ):
        info, rec = self._run_impl(in_filepath, quality)
        info.update(compute_metrics(rec, self._load_img(in_filepath), metrics))
        if return_rec:
            return info, rec
        return info


class PillowCodec(Codec):
    """Abstract codec based on Pillow bindings."""

    fmt = None

    @property
    def name(self):
        raise NotImplementedError()

    @classmethod
    def setup_args(cls, _parser):
        pass

    def _run_impl(self, in_filepath, quality):
        img = self._load_img(in_filepath)
        start = time.time()
        tmp = io.BytesIO()
        img.save(tmp, format=self.fmt, quality=int(quality))
        enc_time = time.time() - start
        tmp.seek(0)
        size = tmp.getbuffer().nbytes

        start = time.time()
        rec = Image.open(tmp)
        rec.load()
        dec_time = time.time() - start

        bpp_val = float(size) * 8 / (img.size[0] * img.size[1])

        out = {
            "bpp": bpp_val,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
        }

        return out, rec


class JPEG(PillowCodec):
    """Use libjpeg linked in Pillow"""

    fmt = "jpeg"
    _description = f"JPEG. Pillow version {PIL.__version__}"

    @property
    def name(self):
        return "JPEG"


class WebP(PillowCodec):
    """Use libwebp linked in Pillow"""

    fmt = "webp"
    _description = f"WebP. Pillow version {PIL.__version__}"

    @property
    def name(self):
        return "WebP"


# class BinaryCodec(Codec):
#     """Call an external binary."""

#     fmt = None

#     @property
#     def name(self):
#         raise NotImplementedError()

#     @classmethod
#     def setup_args(cls, _parser):
#         pass

#     def _run_impl(self, in_filepath, quality):
#         fd0, png_filepath = mkstemp(suffix=".png")
#         fd1, out_filepath = mkstemp(suffix=self.fmt)

#         # Encode
#         start = time.time()
#         run_command(self._get_encode_cmd(in_filepath, quality, out_filepath))
#         enc_time = time.time() - start
#         size = filesize(out_filepath)

#         # Decode
#         start = time.time()
#         run_command(self._get_decode_cmd(out_filepath, png_filepath))
#         dec_time = time.time() - start

#         # Read image
#         rec = read_image(png_filepath)
#         os.close(fd0)
#         os.remove(png_filepath)
#         os.close(fd1)
#         os.remove(out_filepath)

#         img = self._load_img(in_filepath)
#         bpp_val = float(size) * 8 / (img.size[0] * img.size[1])

#         out = {
#             "bpp": bpp_val,
#             "encoding_time": enc_time,
#             "decoding_time": dec_time,
#         }

#         return out, rec

#     def _get_encode_cmd(self, in_filepath, quality, out_filepath):
#         raise NotImplementedError()

#     def _get_decode_cmd(self, out_filepath, rec_filepath):
#         raise NotImplementedError()


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

        total_bpp = 0.0
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

            total_bpp += filesize(compressed) * 8.0 / (H * W)

            start = time.time()
            run_command(self._get_decode_cmd(compressed, raw_out, W, H))
            dec_time += time.time() - start

            recon_int = np.fromfile(raw_out, dtype=np.uint16).reshape(H, W)
            recon_float = recon_int.astype(np.float32) / self.MAX_VAL * drange + cmin
            recons.append(recon_float)

            os.close(fd_out)
            for path in (raw_in, raw_out, compressed):
                os.unlink(path)

        amp_orig  = np.sqrt(I_chan**2  + Q_chan**2)
        amp_recon = np.sqrt(recons[0]**2 + recons[1]**2)
        mse  = float(np.mean((amp_orig - amp_recon) ** 2))
        psnr = 10 * np.log10(amp_orig.max()**2 / (mse + 1e-10))
        bpp = total_bpp / 2 # Bits per pixel per band

        out = {
            "bpp": bpp,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
            "psnr_amp": psnr,
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
    

# class JPEG2000(BinaryCodec):
#     """Use ffmpeg version.
#     (Not built-in support in default Pillow builds)
#     """

#     fmt = ".jp2"

#     @property
#     def name(self):
#         return "JPEG2000"

#     @property
#     def description(self):
#         return f"JPEG2000. ffmpeg version {_get_ffmpeg_version()}"

#     def _get_encode_cmd(self, in_filepath, quality, out_filepath):
#         cmd = [
#             "ffmpeg",
#             "-loglevel",
#             "panic",
#             "-y",
#             "-i",
#             in_filepath,
#             "-vcodec",
#             "jpeg2000",
#             "-pix_fmt",
#             "yuv444p",
#             "-c:v",
#             "libopenjpeg",
#             "-compression_level",
#             quality,
#             out_filepath,
#         ]
#         return cmd

#     def _get_decode_cmd(self, out_filepath, rec_filepath):
#         cmd = ["ffmpeg", "-loglevel", "panic", "-y", "-i", out_filepath, rec_filepath]
#         return cmd


class BPG(BinaryCodec):
    """BPG from Fabrice Bellard."""

    fmt = ".bpg"

    @property
    def name(self):
        return (
            f"BPG {self.bitdepth}b {self.subsampling_mode} {self.encoder} "
            f"{self.color_mode}"
        )

    @property
    def description(self):
        return f"BPG. BPG version {_get_bpg_version(self.encoder_path)}"

    @classmethod
    def setup_args(cls, parser):
        super().setup_args(parser)
        parser.add_argument(
            "-m",
            choices=["420", "444"],
            default="444",
            help="subsampling mode (default: %(default)s)",
        )
        parser.add_argument(
            "-b",
            choices=["8", "10"],
            default="8",
            help="bitdepth (default: %(default)s)",
        )
        parser.add_argument(
            "-c",
            choices=["rgb", "ycbcr"],
            default="ycbcr",
            help="colorspace  (default: %(default)s)",
        )
        parser.add_argument(
            "-e",
            choices=["jctvc", "x265"],
            default="x265",
            help="HEVC implementation (default: %(default)s)",
        )
        parser.add_argument("--encoder-path", default="bpgenc", help="BPG encoder path")
        parser.add_argument("--decoder-path", default="bpgdec", help="BPG decoder path")

    def _set_args(self, args):
        args = super()._set_args(args)
        self.color_mode = args.c
        self.encoder = args.e
        self.subsampling_mode = args.m
        self.bitdepth = args.b
        self.encoder_path = args.encoder_path
        self.decoder_path = args.decoder_path
        return args

    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        if not 0 <= int(quality) <= 51:
            raise ValueError(f"Invalid quality value: {quality} (0,51)")
        cmd = [
            self.encoder_path,
            "-o",
            out_filepath,
            "-q",
            str(quality),
            "-f",
            self.subsampling_mode,
            "-e",
            self.encoder,
            "-c",
            self.color_mode,
            "-b",
            self.bitdepth,
            in_filepath,
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        cmd = [self.decoder_path, "-o", rec_filepath, out_filepath]
        return cmd


class TFCI(BinaryCodec):
    """Tensorflow image compression format from tensorflow/compression"""

    fmt = ".tfci"
    _models = [
        "bmshj2018-factorized-mse",
        "bmshj2018-hyperprior-mse",
        "mbt2018-mean-mse",
    ]

    @property
    def description(self):
        return "TFCI"

    @property
    def name(self):
        return f"{self.model}"

    @classmethod
    def setup_args(cls, parser):
        super().setup_args(parser)
        parser.add_argument(
            "-m",
            "--model",
            choices=cls._models,
            default=cls._models[0],
            help="model architecture (default: %(default)s)",
        )
        parser.add_argument(
            "-p",
            "--path",
            required=True,
            help="tfci python script path (default: %(default)s)",
        )

    def _set_args(self, args):
        args = super()._set_args(args)
        self.model = args.model
        self.tfci_path = args.path
        return args

    def _get_encode_cmd(self, in_filepath, quality, out_filepath):
        if not 1 <= quality <= 8:
            raise ValueError(f"Invalid quality value: {quality} (1, 8)")
        cmd = [
            sys.executable,
            self.tfci_path,
            "compress",
            f"{self.model}-{quality:d}",
            in_filepath,
            out_filepath,
        ]
        return cmd

    def _get_decode_cmd(self, out_filepath, rec_filepath):
        cmd = [sys.executable, self.tfci_path, "decompress", out_filepath, rec_filepath]
        return cmd


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

        total_bpp = 0.0
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

            total_bpp += filesize(out_filepath) * 8.0 / (H * W)
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

        # Amplitude PSNR instead of RGB PSNR
        amp_orig  = np.sqrt(I_chan**2 + Q_chan**2)
        amp_recon = np.sqrt(recons[0]**2 + recons[1]**2)
        mse  = float(np.mean((amp_orig - amp_recon) ** 2))
        psnr = 10 * np.log10(amp_orig.max()**2 / (mse + 1e-10))
        bpp = total_bpp / 2 # Bits per pixel per band

        out = {
            "bpp": bpp,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
            "psnr_amp": psnr,
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

        total_bpp = 0.0
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
                "--InputChromaFormat=420",
                f"--InputBitDepth={self.BITDEPTH}",
                f"--InternalBitDepth={self.BITDEPTH}",
                f"--OutputBitDepth={self.BITDEPTH}",
                "--Profile=main-RExt",
            ]

            start = time.time()
            run_command(cmd)
            enc_time += time.time() - start

            total_bpp += filesize(out_filepath) * 8.0 / (H * W)
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

        # Amplitude PSNR instead of RGB PSNR
        amp_orig  = np.sqrt(I_chan**2 + Q_chan**2)
        amp_recon = np.sqrt(recons[0]**2 + recons[1]**2)
        mse  = float(np.mean((amp_orig - amp_recon) ** 2))
        psnr = 10 * np.log10(amp_orig.max()**2 / (mse + 1e-10))
        bpp = total_bpp / 2 # Bits per pixel per band

        out = {
            "bpp": bpp,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
            "psnr_amp": psnr,
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

        total_bpp = 0.0
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

            total_bpp += filesize(out_filepath) * 8.0 / (H * W)

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

        # Amplitude PSNR instead of RGB PSNR
        amp_orig  = np.sqrt(I_chan**2 + Q_chan**2)
        amp_recon = np.sqrt(recons[0]**2 + recons[1]**2)
        mse  = float(np.mean((amp_orig - amp_recon) ** 2))
        psnr = 10 * np.log10(amp_orig.max()**2 / (mse + 1e-10))
        bpp = total_bpp / 2 # Bits per pixel per band

        out = {
            "bpp": bpp,
            "encoding_time": enc_time,
            "decoding_time": dec_time,
            "psnr_amp": psnr,
        }

        return out, tuple(recons)
