# 0.0008 0.0012 0.0016 0.0024 0.0032 0.004

for lmbda in 0.008 0.016 0.032 0.128; do
    echo "Lambda: ${lmbda} | Alpha: 0.5"
    python train.py --lambda "${lmbda}" -e 200 -bs 32 --dct --dist "l1_ssim" --model_name AHT_DCT-ls-a5
    python test.py --run_name "AHT_DCT-ls-a5_lmbda${lmbda}" --dct
    done
done