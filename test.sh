for lmbda in 0.0012 0.008; do
    echo "Lambda: ${lmbda}"
    python test.py --lambda "${lmbda}" --run_name "AHT_DCT-pconv2_lmbda${lmbda}" --dct -data "/scratch/zb7df/data/NGA/multi_pol/validation"
done
    