ARG BASE=ubuntu:latest
ARG PYTHON=3.11

FROM ${BASE}

ARG BASE
ARG PYTHON

LABEL org.opencontainers.image.source https://github.com/airtai/nbdev-mkdocs

# needed to suppress tons of debconf messages
ENV DEBIAN_FRONTEND noninteractive

# RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt update --fix-missing --yes \
    && apt upgrade --yes \
    && apt install --assume-yes \
      apt-utils git zip curl sudo wget vim fish pip htop nano less wget openssh-client \
      software-properties-common git\
    && apt purge --auto-remove \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*


RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt update --yes \
    && apt install --assume-yes \
      python3.7 \
      python3.8 \
      python3.9 \
      python3.10 \
      python3.11 \
      python3.7-distutils \
      python3.8-distutils \
      python3.9-distutils \
      python3.10-distutils \
      python3.11-distutils \
    && apt purge --auto-remove \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*


RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 4
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 5

RUN update-alternatives --set python3 /usr/bin/python${PYTHON}

RUN python3 -V
RUN python3 -m pip install --upgrade pip
