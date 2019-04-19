FROM tensorflow/tensorflow:1.13.1-gpu-py3

MAINTAINER toanngosy <toanngosy@gmail.com>

ENTRYPOINT [ "/bin/bash", "-c" ]

RUN apt-get update && apt-get install -y \
        bc \
        build-essential \
        cmake \
        curl \
        g++ \
        gfortran \
        git \
        libffi-dev \
        libfreetype6-dev \
        libhdf5-dev \
        libjpeg-dev \
        liblcms2-dev \
        libopenblas-dev \
        liblapack-dev \
        libssl-dev \
        libtiff5-dev \
        libwebp-dev \
        libzmq3-dev \
        nano \
        pkg-config \
        python-dev \
        software-properties-common \
        unzip \
        vim \
        wget \
        zlib1g-dev \
        qt5-default \
        libvtk6-dev \
        zlib1g-dev \
        libjpeg-dev \
        libwebp-dev \
        libpng-dev \
        libtiff5-dev \
        libjasper-dev \
        libopenexr-dev \
        libgdal-dev \
        libdc1394-22-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libtheora-dev \
        libvorbis-dev \
        libxvidcore-dev \
        libx264-dev \
        yasm \
        libopencore-amrnb-dev \
        libopencore-amrwb-dev \
        libv4l-dev \
        libxine2-dev \
        libtbb-dev \
        libeigen3-dev \
        python-dev \
        python-tk \
        python-numpy \
        python3-dev \
        python3-tk \
        python3-numpy \
        ant \
        default-jdk \
        doxygen \
        && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

#Install Anaconda3
RUN wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O anaconda3.sh && \
/bin/bash anaconda3.sh -b -p /opt/conda && \
rm anaconda3.sh

ENV PATH /opt/conda/bin:$PATH

#Install opensim-rl
RUN conda update conda
RUN conda create -n opensim-rl -c kidzik opensim python=3.6.1
RUN echo "source activate opensim-rl" > ~/.bashrc
ENV PATH /opt/conda/envs/opensim-rl/bin:$PATH
RUN conda install -c conda-forge lapack git
RUN pip --no-cache-dir install git+https://github.com/stanfordnmbl/osim-rl.git

EXPOSE 8888 6006 5000

WORKDIR /root


