module load gcc/11.2.0-gcc-8.5.0-cuda-11.2.2-yzmk
module load cmake/3.30.5-gcc-12.4.0-ffna
module load ffmpeg/6.0-gcc-12.4.0-py-3.8.18-jncd

# #################
# ### VVC (VTM) ###
# #################

# cd /home/zb7df/dev/
# # git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
# cd VVCSoftware_VTM
# # git tag | grep VTM-23   # check available tags; pick latest e.g. VTM-23.4
# git checkout VTM-23.14

# mkdir build && cd build
# cmake3 .. -DCMAKE_BUILD_TYPE=Release
# make -j$(nproc)

export LD_LIBRARY_PATH=/opt/ohpc/pub/spack/apps/linux-rocky8-zen/gcc-8.5.0/gcc-11.2.0-yzmkabhif3i3pfhmmxhwkatq2yeq676n/lib64:$LD_LIBRARY_PATH

# #################
# ### HEVC (HM) ###
# #################
# cd /home/zb7df/dev/
# # git clone https://vcgit.hhi.fraunhofer.de/jvet/HM.git
# cd HM
# mkdir build && cd build
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j$(nproc)

#################
### AOM (AV1) ###
#################
# cd /home/zb7df/dev/
# git clone https://aomedia.googlesource.com/aom
# cd ../aom
# mkdir bin && cd bin
# cmake .. -DCMAKE_BUILD_TYPE=Release
# make -j$(nproc)