for lmbda in 0.25 0.5 0.75 1.0 1.25 1.5; do
    for alpha in 0.0017; do
        echo "Lambda: ${lmbda} | Alpha: ${alpha}"
        python train.py --lambda "${lmbda}" --alpha "${alpha}" -e 200 -bs 32 --dct --iq_loss "l1_ssim" --model_name AHT_DCT-pconv4
        python test.py --lambda "${lmbda}" --run_name "AHT_DCT-pconv4_lmbda${lmbda}" --dct
    done
done
