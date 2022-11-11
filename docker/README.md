# DockerFile使用教程s
1.从MMEdu官方构建Docker镜像
1.从GitHub或Gitee中拉取最新的MMEdu项目仓库
```
# 构建默认的 PyTorch 1.8.1，CUDA 10.2 版本镜像
# 如果你希望使用其他版本，请参考下文对 Dockerfile进行修改
git clone https://github.com/OpenXLab-Edu/OpenMMLab-Edu.git
cd OpenMMLab-Edu/
docker build -t MMEdu docker/
```

2.运行对应的Docker容器
```
# 需要调用GPU则增加--gpus all指令
# 需要制定docker容器可使用的共享内存大小则增加 --shm-size=8g指令
# 需要复制则增加-v {DATA_DIR}:目的地址 指令
sudo docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/MMEDu/dataset MMEdu
```

2.从OpenInnoLab平台构建Docker

3.修改DockerFile以自定义环境
```
FROM #基础镜像，一切从这里开始构建
MAINTAINER #镜像是谁写的，名字+邮箱
RUN #镜像构建的时候被需要运行的命令
ADD #步骤，tomcat镜像，这个tomcat压缩包，添加内容
WORKDIR #镜像的挂载目录
VOLUME #挂载的目录
EXPOST #保留端口配置
CMD #指定这个容器启动的时候要运行的命令，只有之后一个会生效，可被替代
ENTRYPOINT #指定这个容器启动的时候要运行的命令，可以追加命令
COPY #类似ADD，将我们文件拷贝到镜像中
ENV #构建的时候设置环境变量
```