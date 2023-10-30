FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev

RUN pip install --no-cache-dir cspot --upgrade

COPY cspot/ /app/
