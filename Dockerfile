FROM tensorflow/tensorflow:1.13.1-gpu-py3

MAINTAINER toanngosy <toanngosy@gmail.com>

ENTRYPOINT [ "/bin/bash", "-c" ]

RUN apt-get update && apt-get install -y \
        bc \
        build-essential \
        curl \
        git \
        nano \
        pkg-config \
        software-properties-common \
        unzip \
        vim \
        wget \
        ant \
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


