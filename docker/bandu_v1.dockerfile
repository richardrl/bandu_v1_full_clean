#FROM ubuntu:bionic as intermediate

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

ENV LDFLAGS=-L/usr/lib/x86_64-linux-gnu/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install -y python3 python3-pip

COPY requirements.txt /root/requirements.txt

WORKDIR /root

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

RUN --mount=type=cache,target=/root/.cache \
    pip3 install -r requirements.txt

# setup entrypoint
COPY ./entrypoint.sh .

ENTRYPOINT ["./entrypoint.sh"]
CMD ["bash"]