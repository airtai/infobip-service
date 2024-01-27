ARG BASE_IMAGE=ubuntu:22.04

FROM $BASE_IMAGE


SHELL ["/bin/bash", "-c"]

# needed to suppress tons of debconf messages
ENV DEBIAN_FRONTEND noninteractive


RUN apt update --fix-missing && apt upgrade --yes \
    && apt install --assume-yes --fix-missing --no-install-recommends \
    python3.10-dev python3-pip python3-venv \
    unattended-upgrades \
    && apt purge --auto-remove --yes && apt clean && rm -rf /var/lib/apt/lists/*

COPY infobip_service infobip_service
COPY pyproject.toml scheduler_requirements.txt scripts/start_service.sh README.md ./

# Install requirements
RUN pip install --no-cache-dir -e ".[dev]"

# Rocketry doesn't supports pydantic v2
#RUN python3 -m venv venv
#RUN venv/bin/pip install --no-cache-dir -e ".[dev]" && venv/bin/pip install --no-cache-dir -r scheduler_requirements.txt

ENTRYPOINT []
CMD [ "/usr/bin/bash", "-c", "./start_service.sh" ]
