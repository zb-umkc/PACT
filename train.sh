for lmbda in 0.5 0.75 1.0; do
    echo "Lambda: ${lmbda}"
    python train.py --lambda "${lmbda}" --alpha 0.0017 -e 200 -bs 32 --dct --iq_loss "l1_ssim" --model_name AHT_losstesting
done