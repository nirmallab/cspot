FROM python:3.9

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir gatorpy --upgrade

COPY gatorpy/ /app/
