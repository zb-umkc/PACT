# 0.0018 0.0035 0.0067 0.013 0.025 0.0483

for lmbda in 0.0009 0.0018 0.0035 0.0067 0.013 0.025 0.0483; do
    echo "Lambda: ${lmbda} | L1 - MS-SSIM | Alpha: 0.5"
    python test.py --run_name "AHT_DCT-l1ssim_lmbda${lmbda}" --dct
done
