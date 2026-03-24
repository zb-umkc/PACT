for lmbda in 0.025; do
    echo "Lambda: ${lmbda}"
    python test.py --lambda "${lmbda}" --run_name "AHT_DCT-pconv4_lmbda${lmbda}" -data "/scratch/zb7df/data/NGA/multi_pol/validation"
done
