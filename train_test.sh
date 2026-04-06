# Lambdas are scaled by 75 to approximate scale difference
# between I/Q L1-SSIM Loss and Amplitude SQNR Loss
lambdas=(
    "0.0008,0.0625"
    "0.0017,0.125"
    "0.0033,0.25"
    "0.005,0.375"
    "0.0067,0.5"
    "0.0083,0.6225"
    "0.01,0.75"
    "0.015,1.125"
)

for pair in "${lambdas[@]}"; do
    IFS=',' read -r lmbda1 lmbda2 <<< "$pair"

    for groups in 8; do

        # Phase 1: Optimize for I/Q only (alpha=1.0)
        echo "PHASE 1: Lambda=${lmbda1} | Groups=${groups}"
        python train.py \
            --lambda "${lmbda1}" \
            --alpha 1.0 \
            -g "${groups}" \
            -e 250 \
            -bs 32 \
            --model_name PACT_g${groups}alpha1.0
        python test.py \
            --lambda "${lmbda1}" \
            -g "${groups}" \
            --run_name "PACT_g${groups}alpha1.0_lmbda${lmbda1}" \
            -data "/scratch/zb7df/data/NGA/multi_pol/test"\

        # Phase 2: Fine-tune for Amplitude only (alpha=0.0)
        echo "PHASE 2: Lambda=${lmbda2} | Groups=${groups}"
        python train.py \
            --lambda "${lmbda2}" \
            --alpha 0.01 \
            -g "${groups}" \
            -e 100 \
            -bs 32 \
            --model_name PACT_g${groups}alpha0.01 \
            --checkpoint "PACT_g${groups}alpha1.0_lmbda${lmbda1}/epoch_best.pth.tar" \
            --learning-rate 1e-4 \
            --reset-lr
            
        python test.py \
            --lambda "${lmbda2}" \
            -g "${groups}" \
            --run_name "PACT_g${groups}alpha0.01_lmbda${lmbda2}" \
            -data "/scratch/zb7df/data/NGA/multi_pol/test"

    done

done
