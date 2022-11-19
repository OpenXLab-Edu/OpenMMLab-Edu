# 完整版安装：容器镜像安装

## 1. Docker 安装

1） 使用对象通过https获取最新的源 

   ```
 $ sudo apt-get update
 $ sudo apt-get -y install apt-transport-https ca-certificates curl software-proper
   ```

2） 安装gpg证书(推荐使用阿里源，若使用华为相关框架/容器则需要修改为中科大源/清华源)​

   ```
 $ curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo apt-key
   ```

3） 设置稳定版仓库

   ```
 $ sudo add-apt-repository "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/l
   ```

4） 更新依赖并安装最新版本docker

   ```
 $ sudo apt-get install docker-ce docker-ce-cli containerd.io
   ```

<!-- 启动docker

```
$ sudo systemctl enable docker
$ sudo systemctl start docker
``` -->

运行hello-world内置镜像测试安装是否成功
```

$ sudo docker run --rm hello-world

```

## 2. 使用MMEdu
### 2.1 操作并管理Docker
#### 查看镜像
```

$ sudo docker images

```
![](2022-10-21-14-14-37.png)

#### 运行镜像
镜像需要在容器中才能运行，运行仓库中的镜像，会自动创建容器。[ctrl + D]可退出并关闭当前容器，若只想退出不想关闭容器请使用[ctrl + P + Q]，建议设置端口映射方便后续使用jupyter notebook
<!-- 每次运行会产生相应的容器 -->
<!-- sudo docker run -it {REPOSITORY:TAG|IMAGE ID}  -->
```

# docker run -it {仓库名：标签|镜像id}, 以下实例均以xedu2.0为仓库名、latest为标签名

$ sudo docker run -it -p 7777:8888 xedu2.0:latest

```
<!-- 进入终端：
```

(base) root@491f14ed3b7e:/# 

```
退出终端：
```

(base) root@491f14ed3b7e:/# exit

``` -->
<!-- ![](2022-10-21-14-17-09.png) -->

#### 管理容器
查看容器相关信息
<!-- 由镜像生成容器，镜像相当于类，容器相当于类创建的对象 -->
```

$ sudo docker ps -a

```
![](2022-10-21-14-25-04.png)

容器命名(根据上一步查看到的容器id)
```

# docker rename 容器id 容器名

$ sudo docker rename de9d447306f0 my_container

```
<!-- 根据容器的改动创建新的仓库/镜像
<!-- 可以将容器保存为新的镜像 -->
<!-- $ sudo docker commit CONTAINER ID  REPOSITORY[:TAG] -->
<!-- ```
$ sudo docker commit CONTAINER ID  REPOSITORY[:TAG]
```

![](2022-10-21-14-27-11.png) -->
进入正在运行的容器

```
$ sudo docker attach my_container
```

重启容器，若使用[ctrl + D]退出了容器，可使用以下命令重启该容器

```
$ sudo docker restart my_container
```

删除容器

```
$ sudo docker rm my_container
```

<!-- 删除镜像 -->
<!-- sudo docker image rm {REPOSITORY:TAG|IMAGE ID} -->
<!-- ```
sudo docker rmi xedu2.0:latest

``` -->
<!-- ![](2022-10-21-14-29-57.png) -->

### 2.2 运行Demo
拷贝文件到docker内
```

# docker cp 本地文件路径 {容器ID|容器名}: 容器内路径

$ sudo docker cp ~/Desktop/demo my_container:home

```
激活环境
```

conda activate xedu

```
安装jupyter notebook
```

pip install jupyter

```
启动notebook
```

jupyter-notebook --port=8888 --allow-root

```
复制终端链接，将8888替换为7777，在浏览器中访问jupyter nootebook。此处端口7777为运行镜像时设定的端口,注意端口匹配
```

localhost/7777/{token}

```
![](2022-10-31-11-46-28.png)
<!-- # VS Code 使用MMEdu教程

1. 在VS Code安装remote ssh，docker的扩展工具​

2. 先在cmd中启动镜像
![](2022-10-21-14-56-54.png)

3. 用ssh连接docker

   1) 查看docker ip
```

   $ ifconfig -a

   ```
   ![](2022-10-21-16-42-26.png)

   ![](2022-10-21-16-48-57.png)

   2) 在3所指窗口输入
   ```

   ssh username@ip

   ```
   3) 选择第一个

   ![](2022-10-21-16-54-59.png)

   4) 连接docker

   ![](2022-10-21-16-56-05.png)

4. 使用docker实现可视化

![](2022-10-21-16-59-01.png) -->

   ```