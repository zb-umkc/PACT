for lmbda in 0.0018 0.0035 0.0067 0.013 0.025 0.0483; do
    echo "Lambda: ${lmbda}"
    python test.py --lambda "${lmbda}" --run_name "AHT_baseline_lmbda${lmbda}_20260123"
done