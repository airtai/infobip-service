ARG BASE_IMAGE=ubuntu:22.04

FROM $BASE_IMAGE


SHELL ["/bin/bash", "-c"]

# needed to suppress tons of debconf messages
ENV DEBIAN_FRONTEND noninteractive


RUN apt update --fix-missing && add-apt-repository ppa:deadsnakes/ppa && apt update \
    && apt upgrade --yes \
    && apt install --assume-yes --fix-missing --no-install-recommends \
    python3.12-dev python3-pip \
    unattended-upgrades \
    && apt purge --auto-remove --yes && apt clean && rm -rf /var/lib/apt/lists/*

COPY infobip_service infobip_service
COPY pyproject.toml scripts/start_service.sh ./

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 2
RUN update-alternatives --config python3

# Install requirements
RUN pip install -e ".[dev]"


ENTRYPOINT []
CMD [ "/usr/bin/bash", "-c", "./start_service.sh" ]
