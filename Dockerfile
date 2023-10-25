FROM tensorflow/tensorflow:latest-gpu

RUN pip install --no-cache-dir cspot --upgrade

COPY cspot/ /app/
