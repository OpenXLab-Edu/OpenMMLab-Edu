from re import T
import mmcv
import os.path as osp
from mmcv import Config
from mmgen.apis import train_model, init_model, sample_img2img_model, sample_unconditional_model
from mmgen.models import build_model
from mmgen.datasets import build_dataset
from mmcv.runner import load_checkpoint
from torchvision import utils
import time
import os


class MMGeneration:
    def __init__(self, 
        backbone='Pix2Pix',
        dataset_path = None,
        tpye = ""
        ):

        # 获取外部运行py的绝对路径
        self.cwd = os.path.dirname(os.getcwd())
        # 获取当前文件的绝对路径
        self.file_dirname = os.path.dirname(os.path.abspath(__file__))

        self.config = os.path.join(self.file_dirname, 'models', 'Pix2Pix/Pix2Pix.py')
        self.checkpoint = os.path.join(self.file_dirname, 'models', '/Pix2Pix/pix2pix_edges2shoes.pth')

        self.backbone = backbone
        backbone_path = os.path.join(self.file_dirname, 'models', self.backbone)        
        ckpt_cfg_list = list(os.listdir(backbone_path))
        for item in ckpt_cfg_list:
            if item[-1] == 'y':
                self.config = os.path.join(backbone_path, item)
            elif item[-1] == 'h':
                self.checkpoint = os.path.join(backbone_path, item)
            else:
                print("Warning!!! There is an unrecognized file in the backbone folder.")
        # self.cfg = Config.fromfile(self.config)
        
        self.dataset_path = dataset_path
        self.lr = None
        self.backbonedict = {
            "Pix2Pix": os.path.join(self.file_dirname, 'models', 'Pix2Pix/Pix2Pix.py'),
            "SinGAN": os.path.join(self.file_dirname, 'models', 'SinGAN/SinGAN.py'),
        }


    def train(self, random_seed=0, save_fold=None, distributed=False, validate=True,
              epoch=50, lr_generators = 0.002, lr_discriminators=0.002, weight_decay=0.001, inverse=False,
              checkpoint = None):

        # 加载网络模型的配置文件
        self.cfg = Config.fromfile(self.backbonedict[self.backbone])
        print("进行了cfg的切换")

        # 如果外部不指定save_fold
        if not self.save_fold:
            # 如果外部也没有传入save_fold，我们使用默认路径
            if not save_fold:
                self.save_fold = os.path.join(self.cwd, 'checkpoints/gen_model')
            # 如果外部传入save_fold，我们使用传入值
            else:
                self.save_fold = save_fold

        if inverse:
            # 如果训练相反任务模型，需要调换source_domain和target_domain，并修改以下的一系列变量
            self.cfg.source_domain, self.cfg.target_domain = self.cfg.target_domain, self.cfg.source_domain
            self.cfg.model.default_domain=self.cfg.target_domain
            self.cfg.model.reachable_domains=[self.cfg.target_domain]
            self.cfg.model.related_domains=[self.cfg.target_domain, self.cfg.source_domain]
            self.cfg.model.gen_auxiliary_loss.data_info=dict(
                pred=f'fake_{self.cfg.target_domain}', 
                target=f'real_{self.cfg.target_domain}'
            )
            self.cfg.custom_hooks[0].res_name_list = [f'fake_{self.cfg.target_domain}']
            self.cfg.evaluation.target_domain = self.cfg.target_domain
                                                
        self.load_dataset(self.dataset_path)
        
        # 创建工作目录
        self.cfg.work_dir = self.save_fold
        mmcv.mkdir_or_exist(osp.abspath(self.cfg.work_dir))

        # 创建模型
        datasets = [build_dataset(self.cfg.data.train)]
        model = build_model(self.cfg.model, train_cfg=self.cfg.train_cfg, test_cfg=self.cfg.test_cfg)
        if not checkpoint:
            model.init_weights()
        else:
            load_checkpoint(model, checkpoint)

        # 根据输入参数更新config文件
        self.cfg.total_iters = epoch * 100
        self.cfg.optimizer.generators.lr = lr_generators          # 生成器的学习率
        self.cfg.optimizer.discriminators.lr = lr_discriminators  # 辨别器的学习率
        # self.cfg.evaluation.metric = metric  # 验证指标
        # self.cfg.runner.max_epochs = epochs  # 最大的训练轮次

        # 设置每 5 个训练批次输出一次日志
        # self.cfg.log_config.interval = 1
        self.cfg.gpu_ids = range(1)
        self.cfg.seed = random_seed
        train_model(
            model,
            datasets,
            self.cfg,
            distributed=distributed,
            validate=validate,
            timestamp=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
            meta=dict()
        )

    def inference(self,
                  is_trained = False,
                  pretrain_model = None,
                  infer_data = None,
                  save_path = "./results/gen_result.png"):

        if not pretrain_model:
            pretrain_model = os.path.join(self.cwd, 'checkpoints/gen_model/latest.pth')

        print("========= begin inference ==========")

        self.save_path = save_path
        checkpoint = self.checkpoint
        checkpoint = os.path.join(self.cwd, 'models', 'checkpoints/gen_model/latest.pth')
        if is_trained:
            # 加载数据集及配置文件的路径
            checkpoint = pretrain_model
            self.load_dataset(self.dataset_path)
        model = init_model(self.cfg, checkpoint, device="cuda:0")

        result = sample_unconditional_model(model, 2, sample_model='orig')
        # result = sample_img2img_model(model, infer_data, self.cfg.target_domain) # 此处的model和外面的无关,纯局部变量
        result = (result[:, [2, 1, 0]] + 1.) / 2.
        # save images
        mmcv.mkdir_or_exist(os.path.dirname(self.save_path))
        utils.save_image(result, self.save_path)

    def load_dataset(self, path):
        self.dataset_path = path
        self.cfg.data.train.dataroot = self.dataset_path
        self.cfg.data.val.dataroot = self.dataset_path
        self.cfg.data.test.dataroot = self.dataset_path