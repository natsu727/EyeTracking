FROM ubuntu:22.04

WORKDIR /home

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update \
	&& apt install -y python3 python3-dev python3-pip python3-setuptools \
	&& apt install -y python3-tk python3-dev cmake libboost-all-dev libgl1-mesa-glx libglib2.0-0 \
	&& apt install -y libxkbcommon-x11-0 libxcb-xinerama0 libqt5gui5

RUN python3 -m pip install --upgrade pip setuptools
RUN python3 -m pip install numpy 
RUN python3 -m pip install opencv-python
RUN pip install dlib
RUN pip install matplotlib==3.7.3
RUN pip install pyautogui==0.9.54

COPY ./ /home/