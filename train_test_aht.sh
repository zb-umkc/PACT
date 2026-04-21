
for lmbda in 0.0018 0.0035 0.0067 0.013 0.025 0.0483; do
    echo "Lambda=${lmbda}"
    # python train.py \
    #     -a AHT \
    #     --lambda "${lmbda}" \
    #     -e 250 \
    #     -bs 32 \
    #     -tr_d "/scratch/zb7df/data/Sandia/train" \
    #     -te_d "/scratch/zb7df/data/Sandia/validation" \
    #     --model_name AHTsandia

    # python test.py \
    #     --lambda "${lmbda}" \
    #     -a AHT \
    #     --run_name "AHTsandia_lmbda${lmbda}" \
    #     -data "/scratch/zb7df/data/Sandia/test"

    # python test.py \
    #     --lambda "${lmbda}" \
    #     -a AHT \
    #     --run_name "AHT_lmbda${lmbda}" \
    #     -data "/scratch/zb7df/data/NGA/multi_pol/validation"

    python test.py \
        --lambda "${lmbda}" \
        -a AHT \
        --run_name "AHT_lmbda${lmbda}" \
        -data "/scratch/zb7df/data/NGA/multi_pol/full"
done
