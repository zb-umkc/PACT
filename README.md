# Helpful Commands
1. Create conda environment (if needed)
    - `conda env create -f environment.yml`
2. Activate conda environment
    - `conda activate pact`
3. Set up SSH for GitHub push/pull access
    - `source ssh.sh`
    - Note: `bash ssh.sh` command will run in sub-shell and not give permissions in outer shell
4. Test model configuration before training
    - `python train.py --size_check [other_args]`
5. Train/test model(s)
    - Modify train_test.sh as needed
    - `bash train_test.sh`
6. Launch Tensorboard
    - Open new terminal
    - `conda activate aht`
    - `tensorboard --logdir /scratch/zb7df/checkpoints/PACT`
7. Run benchmarking
    - `source benchmark_setup.sh`
    - `bash benchmark.sh test` (Use 'test', 'validation', or 'test2')
  
### Important Notes:
- `train_test.sh` script contains 2-stage training process for I/Q and Amp loss
- The NGA dataset used here includes 9,000 training patches, 1,000 validation patches, 1,000 256x256 test patches, and 2 1024x1024 test images. 

### Overleaf Projects:
- TBD
