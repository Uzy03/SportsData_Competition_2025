# === Single-stack image (CUDA 11.6 runtime via wheels) =======================
# PyTorch 1.13.1+cu116 は Python 3.10 が安定
FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PYTHONNOUSERSITE=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /workspace

# 基本ユーティリティ（必要最小限）
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

# 依存インストール（cu116 用インデックスで torch を取得）
COPY requirements.txt /workspace/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# ビルド時にインストール確認（GPU無しでもOK）
RUN python - <<'PY'\nimport torch, numpy\nprint('torch', torch.__version__, 'cuda', torch.version.cuda)\nprint('numpy', numpy.__version__)\nPY

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
