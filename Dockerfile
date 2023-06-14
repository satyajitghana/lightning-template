FROM python:3.7.10-slim-buster

RUN export DEBIAN_FRONTEND=noninteractive \
    && echo "LC_ALL=en_US.UTF-8" >> /etc/environment \
    && echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen \
    && echo "LANG=en_US.UTF-8" > /etc/locale.conf \
    && apt update && apt install -y locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install \
    torch==1.9.0+cpu \
    torchvision==0.10.0+cpu \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && rm -rf /root/.cache/pip

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8
