import numpy as np
import subprocess
import struct
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import tempfile

VTM_ENC = "/home/zb7df/dev/VVCSoftware_VTM/bin/EncoderAppStatic"
VTM_DEC = "/home/zb7df/dev/VVCSoftware_VTM/bin/DecoderAppStatic"
VTM_CFG = "/home/zb7df/dev/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg"  # all-intra config
CUSTOM_CFG = "/home/zb7df/dev/VVCSoftware_VTM/cfg/custom.cfg"

qps = [32]
dataset = "test"
image_paths = list(Path(f"/scratch/zb7df/data/NGA/multi_pol/{dataset}/gt_HH/").glob("*.npy"))

def encode_channel_vtm(channel_f32: np.ndarray, qp: int, tmp_dir: Path):
    """
    channel_f32: H x W float32
    qp: Quantization parameter
    tmp_dir: Temporary storage
    Returns (bpp, recon_f32)
    """
    H, W = channel_f32.shape

    # Normalize to uint16
    cmin, cmax = channel_f32.min(), channel_f32.max()
    norm = ((channel_f32 - cmin) / (cmax - cmin + 1e-9) * 4095).clip(0, 4095).astype(np.uint16)

    # Write raw YUV 4:0:0 (luma only — just the raw uint16 array, big-endian is NOT needed; VTM expects little-endian)
    yuv_in = tmp_dir / "input.yuv"
    yuv_in.write_bytes(norm.tobytes())

    bitstream = tmp_dir / "out.bin"
    yuv_out = tmp_dir / "recon.yuv"
    log_file = tmp_dir / "enc.log"

    enc_cmd = [
        VTM_ENC,
        "-c", VTM_CFG,
        "-i", str(yuv_in),
        "-b", str(bitstream),
        "-o", str(yuv_out),
        "-f", "1",
        "-wdt", str(W),
        "-hgt", str(H),
        "-q", str(qp),
        "-fr", "1",
    ]

    print(' '.join(str(x) for x in enc_cmd))

    with open(log_file, 'w') as lf:
        subprocess.run(enc_cmd, check=True)
        # subprocess.run(enc_cmd, stdout=lf, stderr=lf, check=True)

    # Decode
    dec_cmd = [
        VTM_DEC,
        "-b", str(bitstream),
        "-o", str(yuv_out),
        "-d", "12",
    ]
    subprocess.run(dec_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Read reconstructed
    recon_bytes = yuv_out.read_bytes()
    recon_uint16 = np.frombuffer(recon_bytes, dtype=np.uint16).reshape(H, W)

    # Invert normalization
    recon_f32 = recon_uint16.astype(np.float32) / 4095.0 * (cmax - cmin) + cmin

    # Compute bpp
    num_bits = bitstream.stat().st_size * 8
    bpp = num_bits / (H * W)

    return bpp, recon_f32


def evaluate_image(npy_path: Path, qp: int, tmp_dir: Path):
    data = np.load(npy_path).astype(np.float32)  # shape: (256, 256, 8)
    data_hh = np.stack([data[:,:,0], data[:,:,1]], axis=0) # shape: (2, 256, 256)
    I_chan = data_hh[0]  # HH_I
    Q_chan = data_hh[1]  # HH_Q

    tmp_i = tmp_dir / "I"
    tmp_i.mkdir(parents=True, exist_ok=True)
    tmp_q = tmp_dir / "Q"
    tmp_q.mkdir(parents=True, exist_ok=True)

    bpp_I, recon_I = encode_channel_vtm(I_chan, qp, tmp_i)
    bpp_Q, recon_Q = encode_channel_vtm(Q_chan, qp, tmp_q)

    total_bpp = (bpp_I + bpp_Q) / 2

    # Example metrics (add SSIM, etc. as needed)
    amp_orig  = np.sqrt(I_chan**2  + Q_chan**2)
    amp_recon = np.sqrt(recon_I**2 + recon_Q**2)
    mse  = np.mean((amp_orig - amp_recon)**2)
    psnr = 10 * np.log10(amp_orig.max()**2 / (mse + 1e-10))

    return {"bpp": total_bpp, "psnr": psnr, "mse": mse}


def job(args):
    path, qp = args
    with tempfile.TemporaryDirectory() as tmp:
        return path.stem, qp, evaluate_image(path, qp, Path(tmp))


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=32) as ex:
        results = list(ex.map(job, [(p, q) for p in image_paths for q in qps]))

    print("---- RESULTS ----")
    print(results)