# DRCT用 Dockerfile
# CUDA 12.4とH100に対応したPyTorchベースイメージを使用
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# 非インタラクティブモードに設定（タイムゾーン設定などの対話を回避）
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# 作業用ディレクトリを設定
WORKDIR /workspace

# システムパッケージの更新とOpenCVに必要なライブラリをインストール
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python環境の準備
RUN pip install --upgrade pip

# Python依存関係をインストール
# 元のrequirements.txtがある場合はそれを使用、なければ主要なパッケージを個別インストール
RUN pip install opencv-python numpy scipy matplotlib Pillow tqdm PyYAML tensorboard

# BasicSRをインストール（DRCTの主要依存関係）
RUN pip install basicsr==1.4.2

# 追加で必要になる可能性のあるパッケージ
RUN pip install timm einops

# ポート8888を開放（Jupyter用、必要に応じて）
EXPOSE 8888

# デフォルトコマンド
CMD ["/bin/bash"]