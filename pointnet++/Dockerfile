# Use the tensorflow official image as a parent image.
FROM tensorflow/tensorflow:latest-devel-gpu-py3

# Set the working directory.
WORKDIR /code

# Copy the source code from the project to the working directory
COPY . .

# change the LD LIBRARY PATH
RUN export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:/usr/include/x64_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Instal the tensorflow dependencies
RUN pip3 install --upgrade pip
RUN pip3 install tensorflow

# Compile tensorflow ops
WORKDIR tf_ops
RUN sh compile_ops.sh
