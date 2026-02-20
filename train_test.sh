# 0.0007 0.001 0.0015 0.0025 0.005 0.01

for lmbda in 0.0004 0.0008 0.0016 0.0024 0.0032 0.004; do
    for alpha in 0.5; do
        echo "Lambda: ${lmbda} | Alpha: ${alpha}"
        python train.py --lambda "${lmbda}" --alpha "${alpha}" -e 200 -bs 32 --dct --dist "l1_ssim" --model_name AHT_DCT-ls-a5
        python test.py --run_name "AHT_DCT-ls-a5_lmbda${lmbda}" --dct
    done
done

for lmbda in 0.0004 0.0008 0.0016 0.0024 0.0032 0.004; do
    for alpha in 0.8; do
        echo "Lambda: ${lmbda} | Alpha: ${alpha}"
        python train.py --lambda "${lmbda}" --alpha "${alpha}" -e 200 -bs 32 --dct --dist "l1_ssim" --model_name AHT_DCT-ls-a8
        python test.py --run_name "AHT_DCT-ls-a8_lmbda${lmbda}" --dct
    done
done

for lmbda in 0.0004 0.0008 0.0016 0.0024 0.0032 0.004; do
    for alpha in 0.2; do
        echo "Lambda: ${lmbda} | Alpha: ${alpha}"
        python train.py --lambda "${lmbda}" --alpha "${alpha}" -e 200 -bs 32 --dct --dist "l1_ssim" --model_name AHT_DCT-ls-a2
        python test.py --run_name "AHT_DCT-ls-a2_lmbda${lmbda}" --dct
    done
done