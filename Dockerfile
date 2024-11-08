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
COPY pyproject.toml Makefile ./

RUN python3 -m pip install .[dev]

## if you have zed2i camera, enable following for zed sdk
RUN apt install zstd
ENV ZED_SDK_INSTALLER=ZED_SDK_Tegra_L4T35.3_v4.1.0.zstd.run
RUN wget --quiet -O ${ZED_SDK_INSTALLER} https://download.stereolabs.com/zedsdk/4.1/l4t35.2/jetsons
RUN chmod +x ${ZED_SDK_INSTALLER} && ./${ZED_SDK_INSTALLER} -- silent

WORKDIR /root/libstereosgbm
