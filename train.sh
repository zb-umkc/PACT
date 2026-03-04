for lmbda in 0.25 0.5 1.0 2.0 4.0; do
    echo "Lambda: ${lmbda}"
    python train.py --lambda "${lmbda}" --alpha 0.0017 -e 100 -bs 32 --dct --iq_loss "l1_ssim" --model_name AHT_losstesting
done