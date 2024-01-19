ARG BASE_IMAGE=ubuntu:22.04

FROM $BASE_IMAGE


SHELL ["/bin/bash", "-c"]

# needed to suppress tons of debconf messages
ENV DEBIAN_FRONTEND noninteractive


RUN apt update --fix-missing && apt upgrade --yes \
    && apt install --assume-yes --fix-missing --no-install-recommends \
    python3-dev \
    unattended-upgrades \
    && apt purge --auto-remove --yes && apt clean && rm -rf /var/lib/apt/lists/*

COPY infobip_service infobip_service
COPY pyproject.toml scripts/start_service.sh ./


# Install requirements
RUN pip install -e ".[dev]"


ENTRYPOINT []
CMD [ "/usr/bin/bash", "-c", "./start_service.sh" ]
