for lmbda in 0.0018 0.0035 0.0067 0.013 0.025 0.0483; do
    echo "Lambda: ${lmbda}"
    python train.py --lambda "${lmbda}" -e 200 -bs 32 --dct --model_name AHT_DCT
done