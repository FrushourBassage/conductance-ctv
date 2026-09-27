# The base image pins the CUDA runtime the reported 214 A100 GPU-hours were run
# against. The release has no container-free path to a GPU, so the CPU path is
# the same image with --device cpu.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/conductance-ctv

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md LICENSE NOTICE ./
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

RUN python -m pip install -e .

RUN mkdir -p /opt/conductance-ctv/data /opt/conductance-ctv/runs /opt/conductance-ctv/reports
VOLUME ["/opt/conductance-ctv/data", "/opt/conductance-ctv/reports"]

CMD ["python", "-m", "conductance_ctv.cli.evaluate", "experiment=main"]
