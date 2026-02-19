# 0.0018 0.0035 0.0067 0.013 0.025 0.0483

for lmbda in 0.0067; do
    echo "Lambda: ${lmbda}"
    python train.py --lambda "${lmbda}" -e 3 -bs 32 --dct --model_name AHT_DCT-l1ssim-a5
done