FROM ubuntu:18.04

RUN apt-get update && apt-get install -y \
        software-properties-common
    RUN add-apt-repository ppa:deadsnakes/ppa
    RUN apt-get update && apt-get install -y \
        python3.7 \
        python3-pip
    RUN python3.7 -m pip install pip
    RUN apt-get update && apt-get install -y \
        python3-distutils \
        python3-setuptools
RUN apt-get install -y libgl1-mesa-glx

RUN mkdir -p /usr/src/flask_app/
WORKDIR /usr/src/flask_app/
RUN python3.7 -m pip install --upgrade pip

COPY requirements.txt /usr/src/flask_app/
RUN python3.7 -m pip install -r requirements.txt

COPY . /usr/src/flask_app
# COPY . ./
ENTRYPOINT ["python3.7", "application.py"]
EXPOSE 5000