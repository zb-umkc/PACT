# Lambdas are scaled by 75 to approximate scale difference
# between I/Q L1-SSIM Loss and Amplitude SQNR Loss
# --checkpoint "/scratch/zb7df/models/sar-pact/g16_alpha0.01_dct8x8_N160/nga/lambda_${lmbda2}.pth.tar" \

# For NGA data
lambdas=(
    "0.0008,0.0625"
    "0.0017,0.125"
    "0.0033,0.25"
    "0.005,0.375"
    "0.0067,0.5"
    "0.0083,0.6225"
    "0.01,0.75"
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
    torchrun --nproc_per_node=1 train.py \
        --lambda "${lmbda1}" \
        --alpha 1.0 \
        --gamma 1.0 \
        -g "8" \
        -e 250 \
        -bs 64 \
        --dataset "nga" \
        --run-name "sar-pact/g8_alpha1.0"

    python test.py \
        -a "PACT" \
        --lambda "${lmbda1}" \
        -d "sandia" \
        --split "full" \
        -g "8" \
        --run-name "sar-pact/g8_alpha1.0"

    # Phase 2: Fine-tune for Amplitude only (alpha=0.0)
    echo "PHASE 2: Lambda=${lmbda2} | Groups=8"
    torchrun --nproc_per_node=1 train.py \
        --lambda "${lmbda2}" \
        --alpha 0.01 \
        --gamma 1.0 \
        -g "8" \
        -e 100 \
        -bs 64 \
        --dataset "nga" \
        --checkpoint "/scratch/zb7df/models/sar-pact/g8_alpha1.0/nga/lambda_${lmbda1}.pth.tar" \
        --run-name "sar-pact/g8_alpha0.01"
        
    python test.py \
        -a "PACT" \
        --lambda "${lmbda2}" \
        -d "sandia" \
        --split "full" \
        -g "8" \
        --run-name "sar-pact/g8_alpha0.01"

done