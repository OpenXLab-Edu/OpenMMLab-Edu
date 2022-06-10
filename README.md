# MMEdu

当前版本：0.7

目前支持模块：MMEdu_cls，MMEdu_det，MMEdu_pose，MMEdu_gen, MMEdu_seg, MMEdu_MMBase

## 1.MMEdu是什么？

MMEdu源于国产人工智能视觉（CV）算法集成框架OpenMMLab，是一个“开箱即用”的深度学习开发工具。在继承OpenMMLab强大功能的同时，MMEdu简化了神经网络模型搭建和训练的参数，降低了编程的难度，并实现一键部署编程环境，让初学者通过简洁的代码完成各种SOTA模型（state-of-the-art，指在该项研究任务中目前最好/最先进的模型）的训练，并能够快速搭建出AI应用系统。 

GitHub：https://github.com/OpenXLab-Edu/OpenMMLab-Edu 

国内镜像：https://gitee.com/openxlab-edu/OpenMMLab-Edu

## 2.MMEdu和常见AI框架的比较

### 1）MMEdu和OpenCV的比较

OpenCV是一个开源的计算机视觉框架，MMEdu的核心模块MMCV基于OpenCV，二者联系紧密。

OpenCV虽然是一个很常用的工具，但是普通用户很难在OpenCV的基础上训练自己的分类器。MMEdu则是一个入门门槛很低的深度学习开发工具，借助MMEdu和经典的网络模型，只要拥有一定数量的数据，连小学生都能训练出自己的个性化模型。

### 2）MMEdu和MediaPipe的比较

MediaPipe 是一款由 Google Research 开发并开源的多媒体机器学习模型应用框架，支持人脸识别、手势识别和表情识别等，功能非常强大。MMEdu中的MMPose模块关注的重点也是手势识别，功能类似。但MediaPipe是应用框架，而不是开发框架。换句话说，用MediaPipe只能完成其提供的AI识别功能，没办法训练自己的个性化模型。

### 3）MMEdu和Keras的比较

Keras是一个高层神经网络API，是对Tensorflow、Theano以及CNTK的进一步封装。OpenMMLab和Keras一样，都是为支持快速实验而生。MMEdu则源于OpenMMLab，其语法设计借鉴过Keras。

相当而言，MMEdu的语法比Keras更加简洁，对中小学生来说也更友好。目前MMEdu的底层框架是Pytorch，而Keras的底层是TensorFlow（虽然也有基于Pytorch的Keras）。

### 4）MMEdu和FastAI的比较

FastAI（Fast.ai）最受学生欢迎的MOOC课程平台，也是一个PyTorch的顶层框架。和OpenMMLab的做法一样，为了让新手快速实施深度学习，FastAI团队将知名的SOTA模型封装好供学习者使用。

FastAI同样基于Pytorch，但是和OpenMMLab不同的是，FastAI只能支持GPU。考虑到中小学的基础教育中很难拥有GPU环境，MMEdu特意将OpenMMLab中支持CPU训练的工具筛选出来，供中小学生使用。
