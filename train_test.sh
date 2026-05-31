# Lambdas are scaled by 75 to approximate scale difference
# between I/Q L1-SSIM Loss and Amplitude SQNR Loss

# For NGA data
lambdas=(
    # "0.0008,0.0625"
    # "0.0017,0.125"
    # "0.0033,0.25"
    # "0.005,0.375"
    "0.0067,0.5"
    # "0.0083,0.6225"
    # "0.01,0.75"
)

# For Sandia data
# lambdas=(
    # "0.0008,0.0625"
    # "0.0016,0.125"
    # "0.0033,0.25"
    # "0.0067,0.5"
    # "0.0175,1.3125"
    # "0.03,2.25"
# )

for pair in "${lambdas[@]}"; do
    IFS=',' read -r lmbda1 lmbda2 <<< "$pair"

    # Phase 1: Optimize for I/Q only (alpha=1.0)
    echo "PHASE 1: Lambda=${lmbda1} | Groups=8"
    python train.py \
        --lambda "${lmbda1}" \
        --alpha 1.0 \
        --gamma 0.0 \
        -g "8" \
        -e 250 \
        -bs 32 \
        --dataset "nga" \
        --run-name "sar-pact/g8_alpha1.0_gamma0.0_latentdct"

    python test.py \
        -a "PACT" \
        --lambda "${lmbda1}" \
        -d "nga" \
        --split "full" \
        -g "8" \
        --run-name "sar-pact/g8_alpha1.0_gamma0.0_latentdct"

    # # Phase 2: Fine-tune for Amplitude only (alpha=0.0)
    # echo "PHASE 2: Lambda=${lmbda2} | Groups=8"
    # python train.py \
    #     --lambda "${lmbda2}" \
    #     --alpha 0.01 \
    #     -g "8" \
    #     -e 100 \
    #     -bs 32 \
    #     --train-dataset "/scratch/zb7df/data/Sandia/train" \
    #     --test-dataset "/scratch/zb7df/data/Sandia/validation" \
    #     --model-name PACTsandia_g8alpha0.01 \
    #     --checkpoint "PACTsandia_g8alpha1.0_lmbda${lmbda1}/epoch_best.pth.tar" \
    #     --learning-rate 1e-4 \
    #     --reset-lr
        
    # python test.py \
    #     --lambda "${lmbda2}" \
    #     -d "nga" \
    #     --split "full" \
    #     --run-name "sar-pact/g8_alpha0.01"

done
