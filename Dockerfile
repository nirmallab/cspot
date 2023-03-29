FROM python:3.9

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir gatorpy --upgrade

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

ENV PATH=/opt/conda/bin:$PATH

RUN /opt/conda/bin/conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

RUN conda clean --all --yes

COPY gatorpy/ /app/