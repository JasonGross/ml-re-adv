# Stage 1: Install dependencies
# FROM gcr.io/tpu-pytorch/xla:r2.0 AS builder
FROM python:3.11-bookworm AS builder
RUN pip install --no-cache-dir transformers datasets accelerate torch_xla[tpuvm] torch numpy transformer_lens matplotlib scikit-learn seaborn plotly plotly-express circuitsvis huggingface_hub --upgrade

# Stage 2: Final image
FROM python:3.11-bookworm
# FROM gcr.io/tpu-pytorch/xla:r2.0
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

ENV PJRT_DEVICE=TPU
