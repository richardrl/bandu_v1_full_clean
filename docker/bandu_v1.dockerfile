#FROM ubuntu:bionic as intermediate

#FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV LDFLAGS=-L/usr/lib/x86_64-linux-gnu/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y python3 python3-pip

RUN apt-get install -y python3-dev python3-setuptools

RUN apt-get install -y libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
        libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
        libharfbuzz-dev libfribidi-dev libxcb1-dev

RUN pip3 install pybullet==3.1.7
RUN pip3 install torchvision==0.6.0
RUN pip3 install tqdm==4.46.1
RUN pip3 install wandb==0.10.28
RUN pip3 install numpy==1.18.5
RUN pip3 install deco==0.6.2
RUN pip3 install open3d
RUN pip3 install scipy==1.4.1
RUN pip3 install yacs==0.1.7
RUN pip3 install lxml==4.5.2
RUN pip3 install wandb==0.10.28
RUN pip3 install nflows==0.14
RUN pip3 install torch==1.9.0

# it's important we work with /home/directory after switching users, or else we would have permissions issues
COPY ../requirements.txt /root/requirements.txt

WORKDIR /root

RUN --mount=type=cache,target=/root/.cache \
    pip3 install -r requirements.txt

# use gosu to switch from root to host user ID so we don't run into permissions issues
RUN set -eux; \
	apt-get update; \
	apt-get install -y gosu; \
	rm -rf /var/lib/apt/lists/*; \
# verify that the binary works
	gosu nobody true
# end gosu magic

RUN apt-get update && apt-get install -y tmux

# setup entrypoint
COPY ./entrypoint.sh .

RUN groupadd -r -g 999 docker && useradd -r -g docker -u 999 docker

# these lines are critical. they allow the docker user access to all mounts
# https://stackoverflow.com/questions/23544282/what-is-the-best-way-to-manage-permissions-for-docker-shared-volumes
RUN chown docker:docker /home
RUN mkdir /home/docker
RUN chown docker:docker /home/docker

RUN pip3 install gym
RUN pip3 install urdfpy
RUN pip3 install seaborn
RUN pip3 install trimesh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["bash"]

