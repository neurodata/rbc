FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends && \
    apt-get install -y git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y ssh git man git-annex

RUN pip install numpy scipy datalad pandas tqdm

RUN git clone https://github.com/neurodata/rbc.git && \
    cp -r rbc/* ./

WORKDIR /