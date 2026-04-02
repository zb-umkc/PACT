# Plan:
    # Try (lambda=0.0067, alpha=1.0) for Phase 1 (250 epochs)
    # Then (lambda=1.0, alpha=0.0017) for Phase 2 (~20 epochs? 50 epochs?)
    # Check lr setting when starting Phase 2


# Lambdas are scaled by 150 to approximate scale difference
# between I/Q L1-SSIM Loss and Amplitude SQNR Loss
lambdas=(
    # "0.0033,0.5"
    # "0.005,0.75"
    "0.0067,1.0"
    # "0.0083,1.25"
    # "0.01,1.5"
)

for pair in "${lambdas[@]}"; do
    IFS=',' read -r lmbda1 lmbda2 <<< "$pair"

    for groups in 8; do

        # # Phase 1: Optimize for I/Q only (alpha=1.0)
        # echo "PHASE 1: Lambda=${lmbda1} | Groups=${groups}"
        # python train.py \
        #     --lambda "${lmbda1}" \
        #     --alpha 1.0 \
        #     -g "${groups}" \
        #     -e 250 \
        #     -bs 32 \
        #     --model_name PACT_g${groups}alpha1.0
        # python test.py \
        #     --lambda "${lmbda1}" \
        #     -g "${groups}" \
        #     --run_name "PACT_g${groups}alpha1.0_lmbda${lmbda1}" \
        #     -data "/scratch/zb7df/data/NGA/multi_pol/test"\

        # Phase 2: Fine-tune for Amplitude only (alpha=0.0)
        echo "PHASE 2: Lambda=${lmbda2} | Groups=${groups}"
        python train.py \
            --lambda "${lmbda2}" \
            --alpha 0.1 \
            -g "${groups}" \
            -e 50 \
            -bs 32 \
            --model_name PACT_g${groups}alpha0.1 \
            --checkpoint "PACT_g${groups}alpha1.0_lmbda${lmbda1}/epoch_best.pth.tar" \
            --resume-optimizer \
            --resume-scheduler
        python test.py \
            --lambda "${lmbda2}" \
            -g "${groups}" \
            --run_name "PACT_g${groups}alpha0.0_lmbda${lmbda2}" \
            -data "/scratch/zb7df/data/NGA/multi_pol/test"

    done

done
