FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends && \
    apt-get install -y git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install numpy scipy datalad pandas tqdm

WORKDIR /
