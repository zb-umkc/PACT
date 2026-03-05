for lmbda in 0.5 0.75 1.0 1.25 1.5; do
    echo "Lambda: ${lmbda}"
    python test.py --lambda "${lmbda}" --run_name "AHT_DCT-pconv4_lmbda${lmbda}" --dct -data "/scratch/zb7df/data/NGA/multi_pol/test"
done
