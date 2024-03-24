docker run \
        --runtime=nvidia\
        --gpus all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
        -v /data:/mnt/durable \
        -v /mnt/ephemeral/:/mnt/ephemeral \
        -p 8892:8888 \
        -p 8893:6006 \
        --name \
        kaanan_container \
        kaanan_image
