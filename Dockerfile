FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends && \
    apt-get install -y git wget git-annex && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install numpy scipy datalad pandas tqdm

RUN git clone https://github.com/neurodata/rbc.git && \
    cp -r rbc/* ./

RUN git config --global user.name docker && \
    git config --global user.email docker@docker.docker

WORKDIR /