# 0.5 0.75 1.0 1.25 1.5
for lmbda in 1.0; do
    echo "Lambda: ${lmbda}"
    python train.py --lambda "${lmbda}" --alpha 0.0017 -e 200 -bs 32 --model_name AHT_DCT-gconv${exp}
    python test.py --lambda "${lmbda}" --run_name "AHT_DCT-gconv${exp}_lmbda${lmbda}"
done
