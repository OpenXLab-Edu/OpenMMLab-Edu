# CycleGAN说明文档

> [CycleGAN: Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html)

<!-- [ALGORITHM] -->

## 简介
CycleGAN，即循环生成对抗网络，出自发表于ICCV2017的论文《Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks》，用于图像风格迁移任务。以前的GAN都是单向生成，CycleGAN为了突破Pix2Pix对数据集图片一一对应的限制，采用了双向循环生成的结构，因此得名CycleGAN。

## 网络结构

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/28132635/143049598-23c24d98-7a64-4ab3-a9ba-351db6a0a53d.JPG" />
</div>

## 优点
- 对数据集不再要求一一对应
- 生成图像细节较好

## 适用领域
- 实物图像与其素描像的相互生成
- 语义分割输入与输出图像的相互生成
- 图像昼夜、四季生成
- 其他图片与图片之间的相互生成

## 参考文献

```latex
@inproceedings{zhu2017unpaired,
  title={Unpaired image-to-image translation using cycle-consistent adversarial networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={2223--2232},
  year={2017},
  url={https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html},
}
```
