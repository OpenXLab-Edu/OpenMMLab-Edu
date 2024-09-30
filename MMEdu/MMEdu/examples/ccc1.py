from XEdu.hub import Workflow as wf
import numpy as np

# 模型声明
mm = wf(task='mmedu',checkpoint='ccc1.onnx')
# 待推理图像，此处仅以随机数组为例
image = np.random.random((400,400)) # 可替换成您想要推理的图像路径,如 img = 'cat.jpg'
# 模型推理
res,img = mm.inference(data=image,img_type='cv2')
# 标准化推理结果
result = mm.format_output(lang="zh")
# 可视化结果图像
mm.show(img)

# 更多用法教程详见：https://xedu.readthedocs.io/zh/master/support_resources/model_convert.html