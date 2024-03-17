FROM pytorchlightning/pytorch_lightning:base-cuda-py3.11-torch2.2-cuda12.1.0

RUN apt-get update

RUN mkdir ./kaanan_workspace

COPY ./requirements.txt ./kaanan_workspace/requirements.txt
COPY ./train.py ./kaanan_workspace/train.py
COPY ./config.py ./kaanan_workspace/config.py
COPY ./lrcn.py ./kaanan_workspace/lrcn.py
COPY ./data_module.py ./kaanan_workspace/data_module.py

WORKDIR ./kaanan_workspace
RUN mkdir tensor_board_logs
RUN mkdir confusion_matrices
RUN mkdir final_checkpoints

RUN pip install -r requirements.txt

RUN pip install \
 --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda120

EXPOSE 8888
EXPOSE 6006

CMD jupyter lab --allow-root --ip="*"  --NotebookApp.token='' --NotebookApp.password='' & tensorboard --logdir /kaanan_workspace/tensor_board_logs --bind_all