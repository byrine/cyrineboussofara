# PointNet2

**USAGE**

1-run the tensorflow docker container inside the server:

` docker run --gpus=2 -it tensorflow/tensorflow:latest-devel-gpu-py3`

2-change the LD LIBRARY PATH:

`export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/include/x64_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64`

NB: Otherwise CUDA is broken, then you can check that all the GPU drivers are being deployed in the container with  `nvidia-smi`

3-pip install tensorflow

4-git clone my repository

5-compile the tensorflow Ops with the compilation script in the folder tf_ops:

`sh compile_ops.sh`

NB: This generates the object files:


> tf_sampling_so.so , tf_grouping_so.so , tf_interpolate_so.so


6-run the python code for training Astyx dataset or training ScanNet dataset: 

`python train_astyx.py`

` python train_scannet.py `
