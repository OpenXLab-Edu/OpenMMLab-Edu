ARG PYTORCH="1.8.1"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget python-pyaudio unzip\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# # 在shell中立即激活conda base环境
# SHELL ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate base"]

# 创建并激活一个新的conda虚拟环境
RUN conda create -n xedu_env -y python=3.9 
SHELL ["conda", "run", "-n", "xedu_env", "/bin/bash", "-c"]

# 安装环境 
# RUN pip install unihiker
RUN pip install --upgrade pip wheel setuptools 

RUN pip install  mmcv-full==1.4.5 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8/index.html 

RUN pip install protobuf siot pinpong pandas matplotlib==3.3.1 \
    && pip install jieba pywebio mediapipe==0.10.0 \
    && pip install gradio==3.39.0 seaborn numpy==1.23.0  tqdm yellowbrick \
    && pip install remi pysimplegui pysimpleguiweb pyautogui wordcloud pypinyin \
    && pip install  mmcls==0.20.0 \
    && pip install  matplotlib==3.3.1 \
    && pip install tqdm==4.64.0\
    && pip install opencv-python>=4.1.2.30 \
    && pip install MMEdu --no-deps \
    && pip install basenn --no-deps\
    && pip install baseml basedt basedeploy\
    && pip install --no-cache easy-xedu \
    && pip install mmdet==2.23.0  onnx==1.13.0 onnxsim==0.4.10 onnxruntime==1.13.1 xedu_python jupyterlab

# 暴露端口用于外部通讯
EXPOSE 8888
EXPOSE 5000

COPY activate.sh /xedu/activate.sh

WORKDIR /xedu/
CMD ["bash","activate.sh"]

