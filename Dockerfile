FROM continuumio/miniconda3:22.11.1

RUN pip install --no-cache-dir gatorpy --upgrade

RUN conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

RUN conda clean --all --yes

COPY gatorpy/ /app/
