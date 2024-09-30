# XEdu的安装和下载

[DockerHub](https://hub.docker.com/r/xedu/xedu/tags)

当前在[PyPi](https://pypi.org/user/aiedu/)开源的最新版本号如下：
```
XEdu-python==0.2.3
MMEdu==0.1.28
BaseML==0.1.5
BaseNN==0.3.1
BaseDT==0.1.3
easy-xedu==0.2.2
BaseDeploy==0.0.4
```

## docker容器镜像
--------------

首先需要确保您的电脑系统盘（C盘）空间剩余空间超过5GB，实际建议有10GB及以上空间，便于后续训练使用。如果想要调整存储空间位置，可以参考[这里修改安装路径](https://blog.csdn.net/ber_bai/article/details/120816006)，[这里修改数据路径](https://zhuanlan.zhihu.com/p/410126547)，后文安装过程中也有具体叙述。

### 1.安装Docker软件

这里以Windows11系统（专业版）为例，其他系统可以在网上查找相关教程自行安装Docker，如[菜鸟教程](https://www.runoob.com/docker/windows-docker-install.html)。

Windows11系统中，可以先安装Docker Desktop图形化管理软件，下载链接为：[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)。建议不开启WSL2，否则可能与电脑其他软件存在冲突（除非电脑中已经使用了WSL2虚拟机，那么这里勾选开启）。

![Docker安装](../images/about/docker-install.png)

注：如软件安装空间不足，可以把安装路径指向一个新的路径：可以参考[这里修改安装路径](https://blog.csdn.net/ber_bai/article/details/120816006)

用管理员权限打开CMD，然后输入`mklink /j "C:\Program Files\Docker" "D:\Program Files\Docker"`。这样，软件看似安装在原目录，实则安装在了"D:\Program Files\Docker"。当然也可修改为其他盘。

### 2.启动Docker服务

安装完Docker Desktop，运行启动它，界面如下所示。
![Docker 启动界面](../images/about/docker1.png)
看到左下角显示Engine running说明启动成功。

### 3.拉取镜像

#### 3.1准备工作：检查磁盘剩余存储空间

首先需要检查电脑系统盘（C盘）空间剩余空间是否超过6GB，实际建议有10GB及以上。如果空间足够，可以跳转到[3.2](https://xedu.readthedocs.io/zh/master/about/installation.html#id14)，如空间容器和镜像存储空间不足，旧版本Docker Desktop可以直接在软件中设置新的存储路径，但新版就不行了，下面介绍新版的用法。参考来源：[修改存储路径](https://zhuanlan.zhihu.com/p/410126547)。

##### 1）列出待迁移数据

退出Docker Desktop软件，以防冲突。打开CMD，输入`wsl --list -v`，把所有相关的数据文件列出来，稍后需要挨个迁移。

![Docker 启动界面](../images/about/docker3.1.png)

此时，返回的信息是如上图所示，那么需要迁移的数据有：`docker-desktop-data STOPPED 2`，`docker-desktop STOPPED 2`。有的只出现一条，那么只要迁移这一个就好。接下来，以把数据迁移到D盘为例进行说明。

##### 2）新建保存目录

在D盘新建目录用于保存迁移过去的数据，例如我后续希望相关数据都迁移到"D:\Program Files\Docker"，那么我就得新建这个目录，保证路径"D:\Program Files\Docker"存在。

##### 3）导出数据

在CMD中输入：`wsl --export docker-desktop-data "D:\Program Files\Docker\docker-desktop-data.tar"`。如果有其它要导出，指令类似。例如我们还需要导出`docker-desktop`，那么运行完上一句，继续输入：`wsl --export docker-desktop "D:\Program Files\Docker\docker-desktop.tar"`。

##### 4）注销WSL中原来的数据

在CMD中输入：`wsl --unregister docker-desktop-data`。如果有其它要注销，指令类似。例如我们还需要注销`docker-desktop`，那么运行完上一句，继续输入：`wsl --unregister docker-desktop`。

##### 5）导入数据到新的存储路径

在CMD中输入：`wsl --import docker-desktop-data "D:\Program Files\Docker\data" "D:\Program Files\Docker\docker-desktop-data.tar" --version 2`。这里的"D:\Program Files\Docker\data"是新的存储路径，这个文件夹会自动创建。

若还需要迁移`docker-desktop`，运行完上一句，继续输入：`wsl --import docker-desktop "D:\Program Files\Docker\data" "D:\Program Files\Docker\docker-desktop.tar" --version 2`。

##### 6）重启Docker Desktop

此时已经完成了容器文件的存储位置迁移。如果有问题，可以尝试重启电脑。如果正常迁移完成，可以删除导出的tar文件，即`D:\Program Files\Docker\docker-desktop-data.tar`。如需迁移到其他盘，也可参照此方式完成，只需要修改盘符即可。

#### 3.2拉取镜像

Docker分为容器（Container）和镜像（Image），（有时还会额外有一类叫Dockerfile）。首先需要从云端获取镜像，类似于安装操作系统的镜像，这个镜像是和原版一模一样的。然后可以启动容器，容器可以由用户自主修改。

拉取镜像的命令如下：
`docker pull xedu/xedu:v3s`
打开电脑的命令行（CMD）窗口，输入上面的命令行。

这一步会拉取xedu的镜像文件到本地磁盘，因此务必保证您的电脑系统盘空间剩余空间超过5GB，实际建议有10GB及以上空间，便于后续训练使用。如果想要调整存储空间位置，可以参考上面空间不足的解决办法。刚开始拉取没有相应，可以等待一会儿，就会出现下面的拉取进度的界面。
![Docker拉取界面](../images/about/docker2.png)

等待拉取完成，所用时间取决于网速（大约30分钟-2小时之间），您也可以参考相关教程配置国内镜像源来加快拉取速度。如：[这个办法](https://blog.csdn.net/moluzhui/article/details/132287258)。

### 4.启动docker容器（Container）

在CMD输入：
`docker run -it -p 5000:5000 -p 8888:8888 --mount type=bind,source=D:/share,target=/xedu/share xedu/xedu:v3s`，首次使用会询问是否绑定磁盘，选择Yes。运行成功界面如下：

![Docker Lab](../images/about/docker5.1.png)

接下来就可以用电脑访问 **[127.0.0.1:8888](http://127.0.0.1:8888)** 访问jlab，通过 **[127.0.0.1:5000](http://127.0.0.1:5000)** 访问easytrain。（电脑中的文件想要拷贝进docker，可以放到D盘share文件夹）。美中不足的是，这两个网址需要自行打开浏览器后输入。如果显示效果不佳，可能是浏览器不兼容，建议下载[最新版的chrome浏览器](https://www.google.com/intl/zh-CN/chrome/)。
![Docker Lab](../images/about/docker3.png)
![Docker EasyTrain](../images/about/docker4.png)

#### 可能用到的docker命令
- 查看现有的容器
  `docker ps -a`
- 暂停容器
  `docker stop 34`。
  假设使用ps查看到容器ID是1234567890，还有另一个容器ID是1243567890，我们在指定的时候，只要输入其中的任意一小段，可以区分开不同的容器即可，例如可以用`34`或者`1234`之类来区分这两个不同的容器。
- 再次启动容器
  `docker start 34`
- 进入容器的命令行窗口
  `docker exec 34 -it bash`
### 5.结束容器

在刚才的命令行窗口中，输入CTRL+C，再输入y，即可结束容器。
![Docker shutdown](../images/about/docker5.png)

### 6.重启容器

已完成容器的安装，再次重启容器只需启动Docker服务，再完成5.启动容器的操作即可。

如何快速查看XEdu各模块库的版本
------------------------------

打开python终端，执行以下命令即可查看XEdu各模块库的版本。当前最新版本是0.1.21。

![](../images/mmedu/pip3.png)

注：目前版本MMEdu仅支持CPU。

如何卸载XEdu各模块库
--------------------

如果XEdu某模块库出现异常情况，可以尝试使用`uninstall`命令卸载，然后再使用install命令安装。参考代码：

    $ pip uninstall MMEdu -y
    $ pip uninstall BaseNN -y
    $ pip uninstall BaseML -y
