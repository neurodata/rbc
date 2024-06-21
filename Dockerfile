FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends && \
    apt-get install -y git wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install datalad-installer && \
    datalad-installer git-annex -m datalad/git-annex:release && \
    git config --global filter.annex.process "git-annex filter-process"

RUN pip install numpy scipy datalad pandas tqdm

RUN git clone https://github.com/neurodata/rbc.git && \
    cp -r rbc/* ./

WORKDIR /
