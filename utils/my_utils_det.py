import mmcv
import os.path as osp
from mmcv import Config
from mmdet.apis import inference_model, init_model, show_result_pyplot, train_model
from mmdet.models import build_classifier
from mmdet.datasets import build_dataset
from mmcv.runner import load_checkpoint
import os


class MMDetection:
    def __init__(self, 
        backbone='MobileNet',
        num_classes=-1
        # dataset_type = 'ImageNet'
        ):

        return None


    def train(self, random_seed=0, save_fold='./checkpoints', distributed=False, validate=True, device="cpu",
              metric='accuracy', optimizer="SGD", epochs=100, lr=0.001, weight_decay=0.001, Frozen_stages=1):# 加config

        return None

        
    def inference(self, device='cpu',
                 pretrain_model = './checkpoints/latest.pth',
                 is_trained=False,
                infer_data=None, show=True, neck_iou_threshold=0.99, backbone_iou_threshold=0.99):
        return None


    def load_dataset(self, path, dataset_type):
        return None


# if __name__ == "__main__":
#     img = '../img/test.jpg'
#     # mmcls_test(img)
#     model = MMClassification()
#     result = model.inference(image=img)
#     print(result)
#     show_result_pyplot(model.SOTA_model, img, result)
