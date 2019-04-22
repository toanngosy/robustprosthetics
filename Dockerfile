FROM nvidia/cuda:10.0-base-ubuntu16.04

LABEL maintainer="toanngosy <toanngosy@gmail.com>"

RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        wget \
        && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

#Install Anaconda3
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda3.sh && \
/bin/bash anaconda3.sh -b -p /opt/conda && \
rm anaconda3.sh

ENV PATH /opt/conda/bin:$PATH


EXPOSE 8888 6006 5000

WORKDIR /root
