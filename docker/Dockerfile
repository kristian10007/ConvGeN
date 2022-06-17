
#FROM ubuntu:21.10
FROM nvidia/cuda:11.3.1-cudnn8-devel

RUN ln -s /usr/share/zoneinfo/Europe/Berlin /etc/localtime
RUN apt-get update && apt-get dist-upgrade -y && apt-get install -y python3 pip && apt-get clean

RUN mkdir /benchmark
WORKDIR /benchmark

COPY requirements.txt /benchmark/requirements.txt
RUN pip install -r /benchmark/requirements.txt

COPY run.sh /benchmark/run.sh
WORKDIR /benchmark/data
CMD /benchmark/run.sh
