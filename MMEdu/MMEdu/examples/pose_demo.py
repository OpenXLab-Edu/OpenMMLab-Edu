from MMEdu import MMPose as pose
import shutil
import requests
import time


def only_infer_demo():
    # 人体关键点
    img = 'pose2.jpg' # 指定进行推理的图片路径
    model = pose(backbone='SCNet') # 实例化mmpose模型
    result = model.inference(img=img,device='cpu',show=True) # 在CPU上进行推理
    print(result)

def only_infer_demo1():
    a = time.time()
    img = 'pose.jpg' # 指定进行推理的图片路径
    model = pose(task='body17') # 实例化mmpose模型
    result = model.inference(data=img,device='cpu',checkpoint="rtmpose-m-80e511.onnx") # 在CPU上进行推理
    # rtmpose-m-80e511.onnx
    print(time.time()- a)
    # print(result)
def video_infer_demo():
    import cv2
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("pose.mp4")
    model = pose()
    if not cap.isOpened():
        print("Error opening video file")
    
    while cap.isOpened():
        # a = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        keypoints = model.inference(image=frame,device='cuda',show=False,checkpoint="rtmpose-s-d976b6.onnx") # 在CPU上进行推理
        for j in range(keypoints.shape[0]):
            for i in range(keypoints.shape[1]):
                cv2.circle(frame, (int(keypoints[j][i][0]),int(keypoints[j][i][1])),5,(0,255,0),-1)
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
        # print(time.time()- a)
    cap.release()
    cv2.destroyAllWindows()

def prune():
    import onnx 
    from onnxsim import simplify

    model = onnx.load('face106.onnx')
    model,c= simplify(model)
    print(c)
    onnx.save(model,'face106_p.onnx')

def merge():
    import onnx
    from onnxoptimizer import optimize
    model = onnx.load('body27.onnx')
    passes = ['fuse_bn_into_conv']
    model = optimize(model,passes)

    onnx.save(model,'body27_m.onnx')

if __name__ == "__main__":
    only_infer_demo1()
    # prune()
    # merge()
    # video_infer_demo()
    # download("https://download.openmmlab.com/mmpose/top_down/scnet/scnet50_coco_256x192-6920f829_20200709.pth")
