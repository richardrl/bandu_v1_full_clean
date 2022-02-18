# note: if you have an existing container, this will simply sync into that container and not create
# a new container

IMAGE=richardrl/bandu_v1:latest
#IMAGE=richardrl/bandu_v1:with_root

XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist :0 | sed -e 's/^..../ffff/')
    if [ ! -z "$xauth_list" ]
    then
        echo $xauth_list | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi

docker run -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --volume="$PWD/../:/home/docker/bandu_v1_full_clean":Z \
    --volume="/data/pulkitag/models/rli14/restore_2021-09-12/:/data/pulkitag/models/rli14/restore_2021-09-12":Z \
    --volume="/data/pulkitag/models/rli14/bandu_v1_full_clean/out/datasets/:/data/pulkitag/models/rli14/bandu_v1_full_clean/out/datasets":Z \
    --volume="/data/pulkitag/models/rli14/realsense_docker/:/data/pulkitag/models/rli14/realsense_docker":Z \
    --privileged \
    --runtime=nvidia \
    --net=host \
    -e WANDB_API_KEY \
    -e uid=$(id -u)\
    ${IMAGE} \
    bash