import gradio as gr
import os 
import datetime
from MMEdu import MMClassification as cls
from MMEdu import MMDetection as det
from multiprocessing import Process
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import BaseDeploy as bd
from BaseDT.data import ImageData
det_model_type = det.sota()
cls_model_type = cls.sota()
main_path = os.getcwd()
code = ""
cls_log_save_fold = ""
det_log_save_fold=""
# --------------------function------------------------
# 生成分类训练代码
def generate(model_type,dataset_path,checkpoint, num_classes, epochs, lr):
    # save_fold = os.path.join(main_path, checkpoint)
    # dataset_path = os.path.join(main_path, dataset_path)
#     model_type = model_type.split('-')[1]
    save_fold = checkpoint
    global cls_log_save_fold
    cls_log_save_fold = save_fold
    print('数据集路径加载成功')
    #checkpoint = 'checkpoints/cls_model1/best_accuracy_top-1_epoch_10.pth'   # 请将路径替换为你自己的模型路径
    checkpoint = None # if checkpoint == 'None' else checkpoint
    # if checkpoint != None:
    #     checktemp = checkpoint.split('\\')[-1]  #orange/xx.pth -> xx.pth
    #     checkpoint = os.path.join(main_path,'checkpoints', checkpoint)
    #     save_fold = os.path.join(main_path, 'checkpoints',checktemp.split('.')[0])
    main_code = rf'''from MMEdu import MMClassification as cls
model = cls(backbone='{model_type}')
model.num_classes = {num_classes}
model.load_dataset(path=r'{dataset_path}')
model.save_fold = r'{save_fold}'
model.train(epochs={epochs}, validate=True, device='cpu',lr={lr}, checkpoint={checkpoint})'''
    s=rf'''
```python
{main_code}
```
    '''
    global code 
    code = main_code
    gr.update(visible=True)
    return s,"开始训练"

# 生成分类训练代码
def det_generate(model_type,dataset_path,checkpoint, num_classes, epochs, lr):
    # save_fold = os.path.join(main_path, checkpoint)
    # dataset_path = os.path.join(main_path, dataset_path)
#     model_type = model_type.split('-')[1]
    save_fold = checkpoint
    global det_log_save_fold
    det_log_save_fold = save_fold
    print('数据集路径加载成功')
    #checkpoint = 'checkpoints/cls_model1/best_accuracy_top-1_epoch_10.pth'   # 请将路径替换为你自己的模型路径
    checkpoint = None # if checkpoint == 'None' else checkpoint
    # if checkpoint != None:
    #     checktemp = checkpoint.split('\\')[-1]  #orange/xx.pth -> xx.pth
    #     checkpoint = os.path.join(main_path,'checkpoints', checkpoint)
    #     save_fold = os.path.join(main_path, 'checkpoints',checktemp.split('.')[0])
    main_code = rf'''from MMEdu import MMDetection as det
model = det(backbone='{model_type}')
model.num_classes = {num_classes}
model.load_dataset(path=r'{dataset_path}')
model.save_fold = r'{save_fold}'
model.train(epochs={epochs}, validate=True, device='cpu',lr={lr}, checkpoint={checkpoint})'''
    s=rf'''
```python
{main_code}
```
    '''
    global code 
    code = main_code
    gr.update(visible=True)
    return s,"开始训练"
# 开始训练
def cls_train():
    global code
    print(code)
#     global savefolder_name
#     #从粘贴得到的代码中提取出保存的文件夹名
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    pypath=f'gen_code/train_{formatted_time}.py'
    i=1
    while os.path.exists(pypath):
        pypath=f'gen_code/train_{formatted_time}_{i}.py'
        i+=1

    if not os.path.exists(os.path.dirname(pypath)):
        os.makedirs(os.path.dirname(pypath))

    with open(pypath, 'w') as f:
        f.write(code)
    try:
        def run_gen_code():
            os.system('python '+pypath)
        p = Process(target=run_gen_code)
        p.start()
        p.join()
    except Exception as e:
        print(str(e))
        return str(e)
    plt = draw_loss(cls_log_save_fold)
    log = "训练完成，可以查看图表观察训练情况"
    return log,plt
def det_train():
    global code
    print(code)
#     global savefolder_name
#     #从粘贴得到的代码中提取出保存的文件夹名
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    pypath=f'gen_code/train_{formatted_time}.py'
    i=1
    while os.path.exists(pypath):
        pypath=f'gen_code/train_{formatted_time}_{i}.py'
        i+=1

    if not os.path.exists(os.path.dirname(pypath)):
        os.makedirs(os.path.dirname(pypath))

    with open(pypath, 'w') as f:
        f.write(code)
    try:
        def run_gen_code():
            os.system('python '+pypath)
        p = Process(target=run_gen_code)
        p.start()
        p.join()
    except Exception as e:
        print(str(e))
        return str(e)
    plt = draw_loss(det_log_save_fold)
    log = "训练完成，可以查看图表观察训练情况"
    return log,plt
# 更新按钮为可见
def update_code(a):
    print("update",a)
    gr.Button.update(visible=True,interactive=True)
# 画出loss图
def draw_loss(log_save_fold = "../checkpoints/cls_model/h1"):
    files = os.listdir(log_save_fold)
    json_files = [i  for i in files if 'json' in i ]
    json_files.sort()
    try:
        json_files[-1]
    except Exception as e:
        return e 

    x,y = [],[]
    with open(os.path.join(log_save_fold,json_files[-1])) as f:
        lines = f.readlines()
        for i in lines:
            log_d = eval(i)
            if "mode" in log_d.keys() and log_d['mode']=='train':
                # print(log_d['epoch'],log_d['loss'])
                x.append(log_d['epoch'])
                y.append(log_d['loss'])
    fig = plt.figure()
    plt.plot(x,y)
    # plt.title("Outbreak in " + month)
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    # plt.legend(countries)
    return fig
# 推理
def infer(model_type,checkpoint,img):
    # model_type = model_type.split('-')[1]
    model = cls(backbone=model_type)
    result = model.inference(image=img, show=False, checkpoint=checkpoint)
    # checkpoint_path = os.path.join(main_path,'checkpoints',checkpoint)

    # 从原始权重checkpoint转换为onnx格式model
    if not checkpoint.endswith('.onnx'):
        model_path = checkpoint.replace('.pth', '.onnx')
        model.convert(checkpoint=checkpoint, out_file=model_path)
    else:
        model_path = checkpoint

    image = ImageData(img, backbone=model_type)
    model = bd(model_path)
    result = model.inference(image,show=False)
    infer_code = rf"""
```python
from MMEdu import MMClassification as cls
model = cls(backbon="{model_type}")
img = "{img}"
checkpoint = "{checkpoint}"
result = model.inference(image=img, show=True,checkpoint = checkpoint, device='cpu')
model.print_result(result)
"""
    return infer_code,result,img

def det_infer(model_type,checkpoint,img):
    # model_type = model_type.split('-')[1]
    model = det(backbone=model_type)
    result = model.inference(image=img, show=False, checkpoint=checkpoint)
    # checkpoint_path = os.path.join(main_path,'checkpoints',checkpoint)

    # 从原始权重checkpoint转换为onnx格式model
    if not checkpoint.endswith('.onnx'):
        model_path = checkpoint.replace('.pth', '.onnx')
        model.convert(checkpoint=checkpoint, out_file=model_path)
    else:
        model_path = checkpoint

    image = ImageData(img, backbone=model_type)
    model = bd(model_path)
    result = model.inference(image,show=False)
    infer_code = rf"""
```python
from MMEdu import MMDetection as det 
model = det(backbon="{model_type}")
img = "{img}"
checkpoint = "{checkpoint}"
result = model.inference(image=img, show=True,checkpoint = checkpoint, device='cpu')
model.print_result(result)
"""
    return infer_code,result,img
# ui
with gr.Blocks() as demo:
    with gr.Tab("Classification"):
        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column():
                    gen_input=[
                        gr.Radio(cls_model_type, label="选择模型",value="LeNet"),#info="选择你要训练的模型"
                        gr.Textbox(label="数据集路径",value="../../dataset/cls/hand_gray_q"),
                        gr.Textbox(label="模型权重路径",value="../checkpoints/cls_model/h1"),
                        gr.Textbox(value=3,label="类别数量"),
                        gr.Slider(label="epochs", value=2, minimum=1, maximum=100, step=1),
                        gr.Slider(label="learning rate", value=0.01, minimum=0.001, maximum=2, step=0.001)
                    ]
                    generate_button=gr.Button("生成代码")
                with gr.Column():
                    gen_output = gr.Markdown()

                    train_button = gr.Button("请先点击生成代码")
                    train_output = gr.Markdown()
                    train_loss_plot = gr.Plot()
        with gr.Tab("Inference"):
                    with gr.Row():
                        with gr.Column():
                            infer_gen_input=[
                                gr.Radio(cls_model_type, label="选择模型",value="LeNet"),#info="选择你要训练的模型"
                                gr.Textbox(label="模型权重路径",value="../../checkpoints/cls_model/hand_gray_LeNet/latest.pth"),
                                gr.Textbox(label="上传图片路径",lines=2,value= '/home/user/桌面/pip测试4/dataset/cls/hand_gray/test_set/scissors/testscissors01-02.png'),
                                # gr.Slider(label="epochs", value=2, minimum=1, maximum=100, step=1),
                                # gr.Slider(label="learning rate", value=0.01, minimum=0.001, maximum=2, step=0.001)
                            ]
                            infer_generate_button=gr.Button("生成代码并推理")
                        with gr.Column():
                            inf_outp = [
                                gr.Markdown(),
                                gr.Textbox(label="推理结果"),
                                gr.Image(),
                            ]
    with gr.Tab("Detection"):
        with gr.Tab("Train"):
            with gr.Row():
                with gr.Column():
                    det_gen_input=[
                        gr.Radio(det_model_type, label="选择模型",value="SSD_Lite"),#info="选择你要训练的模型"
                        gr.Textbox(label="数据集路径",value='../../dataset/det/coco'),
                        gr.Textbox(label="模型权重路径",value='../../checkpoints/det_model/plate_gr'),
                        gr.Textbox(value=1,label="类别数量"),
                        gr.Slider(label="epochs", value=2, minimum=1, maximum=100, step=1),
                        gr.Slider(label="learning rate", value=0.01, minimum=0.001, maximum=2, step=0.001)
                    ]
                    det_generate_button=gr.Button("生成代码")
                with gr.Column():
                    det_gen_output = gr.Markdown()

                    det_train_button = gr.Button("请先点击生成代码")
                    det_train_output = gr.Markdown()
                    det_train_loss_plot = gr.Plot()
        with gr.Tab("Inference"):
                    with gr.Row():
                        with gr.Column():
                            det_infer_gen_input=[
                                gr.Radio(det_model_type, label="选择模型",value="SSD_Lite"),#info="选择你要训练的模型"
                                gr.Textbox(label="模型权重路径",value= "../../checkpoints/det_model/pla/best_bbox_mAP_epoch_8.pth"),
                                gr.Textbox(label="上传图片路径",lines=2,value='/home/user/桌面/pip测试9/dataset/det/coco/images/test/0001.jpg'),
                                # gr.Slider(label="epochs", value=2, minimum=1, maximum=100, step=1),
                                # gr.Slider(label="learning rate", value=0.01, minimum=0.001, maximum=2, step=0.001)
                            ]
                            det_infer_generate_button=gr.Button("生成代码并推理")
                        with gr.Column():
                            det_inf_outp = [
                                gr.Markdown(),
                                gr.Textbox(label="推理结果"),
                                gr.Image(),
                            ]


    # cls
    generate_button.click(generate,inputs=gen_input, outputs=[gen_output,train_button])
    # todo: train_button先隐藏起来，点击生成代码后再显示
    # gen_output.change(update_code,inputs=gen_output,outputs=train_button)
    train_button.click(cls_train,outputs=[train_output,train_loss_plot])
    infer_generate_button.click(infer,infer_gen_input, inf_outp)
    # det
    det_generate_button.click(det_generate,inputs=det_gen_input,outputs=[det_gen_output,det_train_button])
    det_train_button.click(det_train,outputs=[det_train_output,det_train_loss_plot])
    det_infer_generate_button.click(det_infer,det_infer_gen_input, det_inf_outp)


demo.launch()
# draw_loss()
# ../../dataset/cls/hand_gray_q
# ../checkpoints/cls_model/h1