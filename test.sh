# 0.0004 0.0008 0.0016 0.0024 0.0032 0.004 0.008 0.016 0.032 0.064 0.128

for lmbda in 0.0004; do
    echo "Lambda: ${lmbda} | L1 - MS-SSIM | Alpha: 0.5"
    python test.py --lambda "${lmbda}" --run_name "AHT_DCT-ls-a5_lmbda${lmbda}" --dct -data "/scratch/zb7df/data/NGA/multi_pol/test"
done
    