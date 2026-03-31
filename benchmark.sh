dataset=$1

### JPEG2000 ###
python -m src.bench sarjpeg2000 \
    /scratch/zb7df/data/NGA/multi_pol/${dataset}/gt_HH/ \
    -q 3,4,6,15 \
    -j 8

### HEVC ###
python -m src.bench sarhm \
    /scratch/zb7df/data/NGA/multi_pol/${dataset}/gt_HH/ \
    -b /home/zb7df/dev/HM/bin \
    -c /home/zb7df/dev/HM/cfg/encoder_intra_main_rext.cfg \
    -q 27,32,37,42 \
    -j 8

### VVC ###
python -m src.bench sarvtm \
    /scratch/zb7df/data/NGA/multi_pol/${dataset}/gt_HH/ \
    -b /home/zb7df/dev/VVCSoftware_VTM/bin \
    -c /home/zb7df/dev/VVCSoftware_VTM/cfg/encoder_intra_vtm.cfg \
    -q 27,32,37,42 \
    -j 8

# ### AV1 ###
# python -m src.bench sarav1 \
#     /scratch/zb7df/data/NGA/multi_pol/${dataset}/gt_HH/ \
#     -b /home/zb7df/dev/aom/bin \
#     -q 27,32,37,42 \
#     -j 8

