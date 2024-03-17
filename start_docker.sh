docker run \
        --runtime=nvidia\
        --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
        -v /mnt:/mnt \
        -v /data/training_data/filelists:/data \
        -p 8892:8888 \
        -p 8893:6006 \
        --name \
        kaanan_container \
        kaanan_image