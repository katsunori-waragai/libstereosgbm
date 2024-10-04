FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

RUN apt-get update
RUN apt install sudo
# only for development
RUN apt update && apt install -y eog nano
RUN apt install -y meshlab


RUN cd /root ; mkdir libstereosgbm
RUN cd /root/libstereosgbm
WORKDIR /root/libstereosgbm
RUN mkdir -p /root/libstereosgbm/stereosgbm/
RUN mkdir /root/libstereosgbm/scripts/
RUN mkdir -p /root/libstereosgbm/test/test-imgs/
COPY stereosgbm/*.py /root/libstereosgbm/stereosgbm/
COPY *.py ./
COPY test/test-imgs/ /root/libstereosgbm/test/test-imgs/
COPY test/*.py test/*.sh /root/libstereosgbm/test/
COPY pyproject.toml ./

RUN python3 -m pip install .[dev]
RUN cd /root ; git clone https://github.com/katsunori-waragai/disparity-view.git
RUN cd /root/disparity-view; python3 -m pip install .[dev]
WORKDIR /root/libstereosgbm
