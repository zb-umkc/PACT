
for lmbda in 0.0018 0.0035 0.0067 0.013 0.025 0.0483; do
    echo "Lambda=${lmbda}"
    # python train.py \
    #     -a AHT \
    #     --lambda "${lmbda}" \
    #     -e 250 \
    #     -bs 32 \
    #     --model_name AHT
    python test.py \
        --lambda "${lmbda}" \
        -a AHT \
        --run_name "AHT_lmbda${lmbda}" \
        -data "/scratch/zb7df/data/NGA/multi_pol/test"
    python test.py \
        --lambda "${lmbda}" \
        -a AHT \
        --run_name "AHT_lmbda${lmbda}" \
        -data "/scratch/zb7df/data/NGA/multi_pol/validation"
done
