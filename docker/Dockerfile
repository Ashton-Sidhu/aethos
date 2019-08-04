FROM ubuntu:18.04

LABEL Name=py_automl Version=0.0.1
EXPOSE 3000

WORKDIR /app
ADD . /app

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y --no-install-recommends && \
  apt-get install -y --no-install-recommends locales && \
  locale-gen en_US.UTF-8 && \
  apt-get install -y --no-install-recommends software-properties-common && \
  apt-get install -y --no-install-recommends \
        make \
        automake \
        libpq-dev \
        libffi-dev \
        gfortran \
        g++ \
        git \
        libboost-program-options-dev \
        libtool \
        libxrender1 \
        wget \
        ca-certificates \
        python3-pip \
        curl && \
  apt-get clean -y && \
  rm -rf /var/lib/apt/lists/*

  # Set environment variables for UTF-8, conda, and shell environments
ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    BASH_ENV=/etc/profile 

RUN python3 -m pip install -r requirements.txt
