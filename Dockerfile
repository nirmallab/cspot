FROM python:3.9

RUN pip install --no-cache-dir gatorpy --upgrade

COPY /gatorpy/* /app/