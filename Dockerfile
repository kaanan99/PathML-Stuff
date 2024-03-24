FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.2-cuda12.1.0

RUN apt-get update

COPY src /workspace/src
COPY requirements.txt /workspace/src

# Set Repo to working directory
WORKDIR /workspace/src

# Create Directories for storing data
RUN mkdir ./tensor_board_logs
RUN mkdir ./confusion_matrices
RUN mkdir ./final_checkpoints

# Install Requirements
RUN pip install -r requirements.txt

RUN pip install \
 --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120

# Expose Ports
EXPOSE 8888
EXPOSE 6006

# Run Jupyter and Tensorboard
CMD jupyter lab --allow-root --ip="*"  --NotebookApp.token='' --NotebookApp.password='' & tensorboard --logdir /kaanan_workspace/tensor_board_logs --bind_all
