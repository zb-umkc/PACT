# Helpful Commands
1. Create conda environment (if needed)
    - `conda env create -f environment.yml`
2. Activate conda environment
    - `conda activate pact`
3. Log into HF CLI (if needed)
    - `hf auth login --token [saved token]`
4. Download datasets (if needed)
    - `hf download --repo-type dataset umkc-mcc/nga --local-dir /scratch/zb7df/data/nga`
    - `hf download --repo-type dataset umkc-mcc/sandia --local-dir /scratch/zb7df/data/sandia`
5. Download models (if needed)
    - `hf download zb-umkc/sar-pact --local-dir /scratch/zb7df/models/sar-pact`
6. Download all of NGA, Sandia, and sar-pact (if needed)
    - `bash hf_download.sh`
7. Set up SSH for GitHub push/pull access
    - `source ssh.sh`
    - Note: `bash ssh.sh` command will run in sub-shell and not give permissions in outer shell
8. Test model configuration before training
    - `python train.py --size-check [other_args]`
9. Train/test model(s)
    - Modify train_test.sh as needed
    - `bash train_test.sh`
10. Launch Tensorboard
    - Open new terminal
    - `conda activate pact`
    - `tensorboard --logdir /scratch/zb7df/logs/sar-pact`
11. Run benchmarking
    - `source benchmark_setup.sh`
    - `bash benchmark.sh test` (Use 'full', 'test', 'validation', or 'test2')
12. Upload models to HF
    - `hf upload zb-umkc/aht /scratch/zb7df/models/aht/benchmarking ./benchmarking`
  
### Important Notes:
- `train_test.sh` script contains 2-stage training process for I/Q and Amp loss
- NGA dataset:
    - 9,000 x (256, 256) training patches
    - 1,000 x (256, 256) validation patches
    - 1,000 x (256, 256) test patches
    - 2 x (1024, 1024) test patches
    - 1 x (3904, 6656) full-res test image
- Sandia dataset:
    - 17,000 x (256, 256) training patches
    - 1,000 x (256, 256) validation patches
    - 1 x (1024, 1024) test patch
    - 1 x (1664, 2560) full-res test image

### Overleaf Projects:
- TBD
