# syntax=docker/dockerfile:1.14
FROM scratch AS training-sources
WORKDIR /app

COPY --chmod=u=rwX,go=rX ml/*.py ml/requirements-training-cpu.txt ml/LICENSE-TinyBERT.txt ml/THIRD-PARTY-intent-v4.txt /app/ml/
COPY --chmod=u=rwX,go=rX ml/data/permission_intent_v2/capability_catalog.json ml/data/permission_intent_v2/contract.json ml/data/permission_intent_v2/representative_instructions.jsonl /app/ml/data/permission_intent_v2/
COPY --chmod=u=rwX,go=rX ml/data/permission_intent_v2/expansion/all_instructions.jsonl ml/data/permission_intent_v2/expansion/contract.json ml/data/permission_intent_v2/expansion/ai_review_report.json ml/data/permission_intent_v2/expansion/split_manifest.json /app/ml/data/permission_intent_v2/expansion/
COPY --chmod=u=rwX,go=rX ml/data/permission_intent_v2/expansion/shards/*.jsonl /app/ml/data/permission_intent_v2/expansion/shards/
COPY --chmod=u=rwX,go=rX ml/data/permission_intent_v2/expansion/splits/*.jsonl /app/ml/data/permission_intent_v2/expansion/splits/
COPY --chmod=u=rwX,go=rX ml/data/permission_intent_v3/*.json ml/data/permission_intent_v3/*.jsonl /app/ml/data/permission_intent_v3/
COPY --chmod=u=rwX,go=rX ml/data/permission_intent_v3/shards/*.jsonl /app/ml/data/permission_intent_v3/shards/
COPY --chmod=u=rwX,go=rX ml/data/permission_intent_v4/*.json ml/data/permission_intent_v4/*.jsonl /app/ml/data/permission_intent_v4/
COPY --chmod=u=rwX,go=rX ml/data/permission_intent_v5/*.json ml/data/permission_intent_v5/*.jsonl ml/data/permission_intent_v5/*.txt /app/ml/data/permission_intent_v5/
COPY --chmod=u=rwX,go=rX ml/data/permission_intent_v6/*.json ml/data/permission_intent_v6/*.jsonl ml/data/permission_intent_v6/*.txt /app/ml/data/permission_intent_v6/
COPY --chmod=u=rwX,go=rX ml/results/tinybert-intent-v2-20261118/data_manifest.json /app/ml/results/tinybert-intent-v2-20261118/
COPY --chmod=u=rwX,go=rX ml/results/tinybert-intent-v3-20261118/evaluation/comparison_report.json /app/ml/results/tinybert-intent-v3-20261118/evaluation/
COPY --chmod=u=rwX,go=rX ml/results/tinybert-intent-v4-20261118/evaluation/comparison_report.json /app/ml/results/tinybert-intent-v4-20261118/evaluation/
COPY --chmod=u=rwX,go=rX tests/test_intent_training.py tests/test_permission_evidence.py /app/tests/
COPY --chmod=u=rwX,go=rX docs/tinybert-intent-v4-training.md docs/tinybert-local-training.md /app/docs/
COPY --chmod=u=rwX,go=rX docker/tinybert-training/entrypoint.py /app/docker/tinybert-training/

# The CUDA-enabled torch wheel includes its user-space CUDA runtime. The
# NVIDIA Container Toolkit supplies the host driver and GPU to the container.
FROM python:3.10.22-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370 AS training
ARG TARGETARCH
RUN test "$TARGETARCH" = amd64 || (echo 'TinyBERT CUDA training requires linux/amd64' >&2 && exit 1)

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/huggingface \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --chmod=u=rwX,go=rX docker/tinybert-training/requirements-cu128.txt /tmp/requirements-cu128.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: -r /tmp/requirements-cu128.txt \
    && python -m pip check \
    && python -c 'import torch, transformers, onnx, onnxruntime; assert torch.version.cuda == "12.8"'

COPY --from=training-sources /app /app
WORKDIR /app
USER 1000:1000
ENTRYPOINT ["python", "/app/docker/tinybert-training/entrypoint.py"]
CMD ["--check"]
