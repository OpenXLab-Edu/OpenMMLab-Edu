import mmcv
from mmcv import Config
import mmedit
from mmedit.apis import matting_inference, init_model
from mmedit.datasets import build_dataset
from mmedit.models import build_model
from mmedit.apis import train_model
import cv2
import os
import tempfile
import os.path as osp
from tqdm import tqdm
from matplotlib import pyplot as plt

class MMMat:
    def __init__(self, 
        backbone = 'IndexNet', 
        dataset_path = None
        ):
        # 获取外部运行py的绝对路径
        self.cwd = os.path.dirname(os.getcwd())
        # 获取当前文件的绝对路径
        self.save_fold = None
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))

        self.backbone = backbone
        backbone_path = os.path.join(self.file_dirname, 'models', self.backbone)
        ckpt_cfg_list = list(os.listdir(backbone_path))
        for item in ckpt_cfg_list:
            if item[-1] == 'y':
                self.config = os.path.join(backbone_path, item)
            elif item[-1] == 'h':
                self.checkpoint = os.path.join(backbone_path, item)
        self.cfg = Config.fromfile(self.config)
        
        self.dataset_path = dataset_path
        self.cfg.load_from = self.checkpoint

        return None


    def train(self, random_seed=0, save_fold=None, checkpoint = None, distributed=False, validate=True,
              metric=['SAD', 'MSE', 'GRAD', 'CONN'],
              optimizer="Adam", epochs=5, lr=2.5e-3, 
              resume_from = None,
              eval_interval = 4,
              log_interval = 5,
              ):

    ##### 为了和其他保持一致使用 epochs （实际是iter）
        print("========= begin training ==========")
        # 如果外部不指定save_fold
        if not self.save_fold:
            # 如果外部也没有传入save_fold，我们使用默认路径
            if not save_fold:
                self.save_fold = os.path.join(self.cwd, 'checkpoints/edit_model')
            # 如果外部传入save_fold，我们使用传入值
            else:
                self.save_fold = save_fold

        # self.cfg = Config.fromfile(self.backbonedict[self.backbone])
        # print(self.cfg.pretty_text)
        #########  区别其他，gpus_id (list) 换成了gpus （int）
        self.cfg.gpus = 1
        self.cfg.work_dir = self.save_fold

        # 如果不使用传入的checkpoint，默认使用网上的pretrain模型
        if checkpoint:
            self.cfg.load_from = checkpoint

        self.cfg.resume_from = resume_from
        self.cfg.seed = random_seed

        # Use smaller batch size for training
        self.cfg.data.train_dataloader.samples_per_gpu = 4
        self.cfg.data.workers_per_gpu = 1

        # The original learning rate (LR) is set for batch size 16 with 1 GPU.
        # We reduce the lr by a factor of 4 since we reduce the batch size.

        #########  区别其他，将optimizer换成了optimizers
        self.cfg.optimizers.lr = lr  # 学习率
        self.cfg.optimizers.type = optimizer  # 优化器
        self.cfg.total_iters = epochs
        self.cfg.lr_config = None



        self.cfg.evaluation.interval = eval_interval
        self.cfg.checkpoint_config.interval = eval_interval

        ########## metric 位置不同  test_cfg.metrics
        self.cfg.test_cfg.metrics = metric  # 验证指标
        # set log interval
        self.cfg.log_config.interval = log_interval

        datasets = [build_dataset(self.cfg.data.train)]

        # build model
        model = build_model(self.cfg.model, train_cfg=self.cfg.train_cfg, test_cfg=self.cfg.test_cfg)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))
        # train model
        train_model(
            model, datasets, self.cfg, distributed=distributed, validate=validate, meta=dict())
        print("========= finish training ==========")
        return None

    
    def inference(self,
                  device='cuda:0',
                  is_trained=False,
                  pretrain_model='./checkpoints/edit_model/latest.pth',
                  merged_path=None, trimap_path=None,
                  img=None,
                  show=True,
                  work_dir='./img_result/',
                  name='mat_result'):
        """
        params:
            device: 推理设备,可选参数: ('cuda:int','cpu')
            is_trained: 是否使用本地预训练的其他模型进行训练
            pretrain_model: 如果使用其他模型，则传入模型路径
            img: 推理图片的路径或文件夹名
            show: 是否对推理结果进行显示
            work_dir: 推理结果图片的保存文件夹
            name: 推理结果保存的名字
        return:
        """

        if not pretrain_model:
            pretrain_model = os.path.join(self.cwd, 'checkpoints/edit_model/latest.pth')
        print("========= begin inference ==========")

        if is_trained == True:
            self.checkpoint = pretrain_model

        # initialize pose model
        model = init_model(self.config, self.checkpoint, device=device)
        pred_alpha = matting_inference(model, merged_path, trimap_path) * 255
        if name:
            plt.gcf().dpi = 80
            plt.axis('off')
            plt.imshow(pred_alpha, cmap=plt.get_cmap('gray'))
            save_path = work_dir+name+'.png'
            print('Image result is save as %s' % (save_path))
            plt.savefig(save_path)
        return pred_alpha

    def load_dataset(self, path):

        self.dataset_path = path

        #数据集修正为 images train.json val.json 形式
        # cfg.data_root = 'data/coco_tiny'
        self.cfg.data.train.type = 'AdobeComp1kDataset'
        self.cfg.data.train.ann_file = os.path.join(self.dataset_path, 'training_list.json')
        self.cfg.data.train.data_prefix = self.dataset_path

        self.cfg.data.val.type = 'AdobeComp1kDataset'
        self.cfg.data.val.ann_file = os.path.join(self.dataset_path,'test_list.json')
        self.cfg.data.val.data_prefix = self.dataset_path

        self.cfg.data.test.type = 'AdobeComp1kDataset'
        self.cfg.data.test.ann_file = os.path.join(self.dataset_path,'test_list.json')
        self.cfg.data.test.data_prefix = self.dataset_path