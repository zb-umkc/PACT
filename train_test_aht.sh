# 0.0018 0.0035 0.0067 0.013 0.025 0.0483

for lmbda in 0.0018 0.0035 0.0067 0.013 0.025 0.0483; do
    echo "Lambda=${lmbda}"
    python train.py \
        -a AHT \
        --lambda "${lmbda}" \
        --alpha 1.0 \
        --iq-loss mse \
        -e 250 \
        -bs 32 \
        --dataset "sandia" \
        --run-name "aht/benchmarking"

    # python test.py \
    #     --lambda "${lmbda}" \
    #     -a AHT \
    #     --run_name "AHTsandia_lmbda${lmbda}" \
    #     -data "/scratch/zb7df/data/Sandia/test"

    # python test.py \
    #     --lambda "${lmbda}" \
    #     -a AHT \
    #     --run_name "AHTsandia_lmbda${lmbda}" \
    #     -data "/scratch/zb7df/data/Sandia/validation"

    python test.py \
        --lambda "${lmbda}" \
        -a AHT \
        -d "sandia" \
        --split "full" \
        --run-name "aht/benchmarking"
done
