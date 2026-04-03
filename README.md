# Instructions for testing on UMKC SSE Cluster
1. Create conda environment (if needed)
    - `conda env create -f environment.yml`
2. Activate conda environment
    - `conda activate aht`
3. Set up SSH for GitHub push/pull access
    - `source ssh.sh`
    - Note: `bash ssh.sh` command will run in sub-shell and not give permissions in outer shell
4. Test model configuration before training
    - `python train.py --size_check [other_args]`
5. Train model(s)
    - Modify train.sh as needed
    - `bash train.sh`
6. Launch Tensorboard
    - Open new terminal
    - `conda activate aht`
    - `tensorboard --logdir /scratch/zb7df/checkpoints/PACT`
7. Run test(s)
    - Modify test.sh as needed
    - `bash test.sh`
8. Run benchmarking
    - `source benchmark_setup.sh`
    - `bash benchmark.sh test` (Use 'test', 'validation', or 'test2')
  
### Important Notes:
- `train_test.sh` script can also be used to train and test each variation in a single command
- `--dct` flag adds DCT and Inverse DCT layers to architecture (AHT-DCT)
- `--dist` argument determines I/Q distortion loss used during training (either MSE or L1-SSIM loss)
- `--alpha` is the weight parameter for L1-SSIM loss (default 0.5)
- The NGA dataset I was given included 10,000 training patches, 1,000 validation patches, and 1 high-res test image. My ELIC-DCT work instead uses the following:
    - 9,000 training patches (from train)
    - 1,000 validation patches (from train)
    - 1,000 testing patches (from validation)
    - The code creating these splits is located at the end of the elic_dct_rle.ipynb notebook.

### Overleaf Projects:
- TBD
