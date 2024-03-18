FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.2-cuda12.1.0

RUN apt-get update

# Clone the Repo
RUN git clone https://github.com/kaanan99/PathML-Stuff.git

# Set Repo to working directory
WORKDIR ./PathML-Stuff 

# Create Directories for storing data
RUN mkdir ./src/tensor_board_logs
RUN mkdir ./src/confusion_matrices
RUN mkdir ./src/final_checkpoints

# Install Requirements
RUN pip install -r requirements.txt

RUN pip install \
 --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120

# Expose Ports
EXPOSE 8888
EXPOSE 6006

# Run Jupyter and Tensorboard
CMD jupyter lab --allow-root --ip="*"  --NotebookApp.token='' --NotebookApp.password='' & tensorboard --logdir /kaanan_workspace/tensor_board_logs --bind_all