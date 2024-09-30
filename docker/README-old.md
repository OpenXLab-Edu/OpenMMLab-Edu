# DockerFile使用教程
## 如果需要安装docker服务，详细请参考docker/start_docker.md文件
## 如果需要在docker内使用GPU相关功能，详细请参考docker/docker_with_gpu.md


## 1.从MMEdu官方构建Docker镜像
### 1.从GitHub或Gitee中拉取最新的MMEdu项目仓库
```
# 构建默认的 PyTorch 1.8.1，CUDA 10.2 版本镜像
# 如果你希望使用其他版本，请参考下文对 Dockerfile进行修改
git clone https://github.com/OpenXLab-Edu/OpenMMLab-Edu.git
cd OpenMMLab-Edu/
sudo docker build -t mmedu docker/
```

### 2.运行对应的Docker容器
```
# 需要调用GPU则增加--gpus all指令
# 需要制定docker容器可使用的共享内存大小则增加 --shm-size=8g指令
# 需要复制则增加-v {DATA_DIR}:目的地址 指令
sudo docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/MMEDu/dataset mmedu
```

## 2.从OpenInnoLab平台构建Docker
### 待更新


## 3.修改DockerFile以自定义环境
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


## 4.TODO List
```
1.docker自启动jupyter并开启端口映射
2.docker中安装onnxruntime并测试
3.docker内调用GPU
4.在虚拟机中进行docker的安装并测试通过
```

## 5.常见错误集合

### 1.拉取OpenMMLab-Edu时http-timeout-Error
```
#  将GitHub的地址更换为gitee地址
git clone https://gitee.com/openxlab-edu/OpenMMLab-Edu.git
```

### 2.docker构建时相关包下载速度太慢
```
# answer: 修改dockerfile中的run命令，换用清华源
RUN pip install --no-cache-dir mmcv-full==1.4.6 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8/index.html -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3.docker换源
```
需要注册阿里云账号并在DockerFile中更改镜像地址为你在阿里云中的地址。
详细内容请参考https://blog.csdn.net/rothchil/article/details/125622078?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522166884841616800213065624%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=166884841616800213065624&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-125622078-null-null.142^v65^control,201^v3^control_2,213^v2^t3_esquery_v3&utm_term=docker换源&spm=1018.2226.3001.4187
```

### 4.Ubuntu apt install换源
在命令行中输入如下命令
```
cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudovim /etc/apt/sources.list
```
在文本中加入如下内容
```
deb http://mirrors.ustc.edu.cn/ubuntu/ precise-updates main restricted
deb-src http://mirrors.ustc.edu.cn/ubuntu/ precise-updates main restricted
deb http://mirrors.ustc.edu.cn/ubuntu/ precise universe
deb-src http://mirrors.ustc.edu.cn/ubuntu/ precise universe
deb http://mirrors.ustc.edu.cn/ubuntu/ precise-updates universe
deb-src http://mirrors.ustc.edu.cn/ubuntu/ precise-updates universe
deb http://mirrors.ustc.edu.cn/ubuntu/ precise multiverse
deb-src http://mirrors.ustc.edu.cn/ubuntu/ precise multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ precise-updates multiverse
deb-src http://mirrors.ustc.edu.cn/ubuntu/ precise-updates multiverse
deb http://mirrors.ustc.edu.cn/ubuntu/ precise-backports main restricted universe multiverse
deb-src http://mirrors.ustc.edu.cn/ubuntu/ precise-backports main restricted universe multiverse
```
更新后即可提速
```
sudo apt-get update
sudo apt-get upgrade
```



### 5.MMEdu Import Error
```
>>> import MMEdu
No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
/opt/conda/lib/python3.7/site-packages/mmcv/cnn/bricks/transformer.py:33: UserWarning: Fail to import ``MultiScaleDeformableAttention`` from ``mmcv.ops.multi_scale_deform_attn``, You should install ``mmcv-full`` if you need this module. 
  warnings.warn('Fail to import ``MultiScaleDeformableAttention`` from '
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/opt/conda/lib/python3.7/site-packages/MMEdu/__init__.py", line 2, in <module>
    from .Detection import MMDetection
  File "/opt/conda/lib/python3.7/site-packages/MMEdu/Detection/__init__.py", line 1, in <module>
    from .Detection_Edu import MMDetection
  File "/opt/conda/lib/python3.7/site-packages/MMEdu/Detection/Detection_Edu.py", line 6, in <module>
    from mmdet.apis import inference_detector, init_detector, show_result_pyplot, train_detector
  File "/opt/conda/lib/python3.7/site-packages/mmdet/apis/__init__.py", line 2, in <module>
    from .inference import (async_inference_detector, inference_detector,
  File "/opt/conda/lib/python3.7/site-packages/mmdet/apis/inference.py", line 7, in <module>
    from mmcv.ops import RoIPool
  File "/opt/conda/lib/python3.7/site-packages/mmcv/ops/__init__.py", line 2, in <module>
    from .active_rotated_filter import active_rotated_filter
  File "/opt/conda/lib/python3.7/site-packages/mmcv/ops/active_rotated_filter.py", line 10, in <module>
    ['active_rotated_filter_forward', 'active_rotated_filter_backward'])
  File "/opt/conda/lib/python3.7/site-packages/mmcv/utils/ext_loader.py", line 13, in load_ext
    ext = importlib.import_module('mmcv.' + name)
  File "/opt/conda/lib/python3.7/importlib/__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ImportError: libcudart.so.10.1: cannot open shared object file: No such file or directory
```

```
# answer: 1.确认dockerfile使用的cuda、torc、mmcv版本符合硬件，如Nvidia-30x显卡无法使用cuda10.x
# answer: 2.在docker环境内的命令行中输入sudo ldconfig /usr/local/cuda-10.2/lib64即可修复
```

