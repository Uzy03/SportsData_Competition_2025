# ベースイメージ
FROM python:3.11-slim-bullseye

# OSパッケージのアップデート
RUN apt-get update && apt-get upgrade -y && apt-get clean

# 必要なパッケージのみインストール
# RUN apt-get install -y <必要なパッケージ>

# 作業ディレクトリの作成
WORKDIR /workspace

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtのコピーとインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードのコピー
COPY . /workspace

# デフォルトコマンド
# CMD ["python", "Player2Vec/main.py"]
