for lmbda in 0.0035 0.0067 0.013 0.025 0.0483; do
    echo "Lambda: ${lmbda}"
    python train.py --lambda "${lmbda}" -e 200 -bs 32 --dct --model_name AHT_DCT-exp2
    python test.py --run_name "AHT_DCT-exp2_lmbda${lmbda}_20260126" --dct
done