# Docker调用GPU官方教程
## Nvidia-Docker安装需要安装两个部分，Docker-CE和NVIDIA Container Toolkit。
### 1.安装docker-ce(若已经安装过则跳过此步骤)
```
curl https://get.docker.com | sh && sudo systemctl --now enable docker
```
安装完后需要执行
```
sudo /lib/systemd/systemd-sysv-install enable docker
```
来激活docker，激活后查看Docker版本
```
docker --version
```

### 2.安装NVIDIA Container Toolkit
设置稳定版本的库及GPG密钥
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
可选部分：如果想要使用实验特性，需要加入experimental分支到库下：
```
curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
```
更新好包列表之后，安装nvidia-docker2包及其依赖：
```
sudo apt-get update
sudo apt-get install -y nvidia-docker2
```

上述默认运行时设置好后，重启Docker后台驻留程序：
```
sudo systemctl restart docker
```


### 3.测试
现在可以通过运行base CUDA container来测试一个working setup
```
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```
