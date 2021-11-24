FROM ubuntu:bionic

RUN apt-get update

RUN apt-get install -y python3 python3-pip

COPY requirements.txt /root/requirements.txt

WORKDIR /root

ENV LDFLAGS=-L/usr/lib/x86_64-linux-gnu/

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y python3-dev python3-setuptools

RUN apt-get install -y libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
        libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
        libharfbuzz-dev libfribidi-dev libxcb1-dev

RUN --mount=type=cache,target=/root/.cache \
    pip3 install -r requirements.txt

