import torch
import numpy as np
import os

from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.engine.hooks import HookBase
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_train_loader
from tqdm import tqdm

import logging
logging.getLogger("detectron2").setLevel(logging.INFO)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

base_path = '.'

register_coco_instances("my_dataset_train", {}, f'{base_path}/annotations/COCO/train.json', f'{base_path}/dataset/images/')
register_coco_instances("my_dataset_val", {}, f'{base_path}/annotations/COCO/val.json', f'{base_path}/dataset/images/')
train_metadata = MetadataCatalog.get("my_dataset_train")
train_dataset_dicts = DatasetCatalog.get("my_dataset_train")
val_metadata = MetadataCatalog.get("my_dataset_val")
val_dataset_dicts = DatasetCatalog.get("my_dataset_val")
num_train_samples=len(train_dataset_dicts)
num_val_samples=len(val_dataset_dicts)

batch_size = 4
patience = 10
num_train_steps = num_train_samples//batch_size
num_val_steps = num_val_samples//batch_size

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.OUTPUT_DIR = './Saved Models/Detectron2_models/'
cfg.DATASETS.TRAIN = ("my_dataset_train")
cfg.DATASETS.VAL = ("my_dataset_val",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.INPUT.MIN_SIZE = 640
cfg.INPUT.MIN_SIZE_TRAIN = 640
cfg.INPUT.MIN_SIZE_TEST = 640
cfg.INPUT.MAX_SIZE = 640
cfg.INPUT.MAX_SIZE_TRAIN = 640
cfg.INPUT.MAX_SIZE_TEST = 640
cfg.DATALOADER.NUM_WORKERS = 0
cfg.SOLVER.IMS_PER_BATCH = batch_size
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 300*num_train_steps   
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        train_augmentations = [
            T.ResizeShortestEdge(short_edge_length=[640,640], max_size=640),
            T.RandomBrightness(0.4, 1.6),
            T.RandomSaturation(0.3, 1.7),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        ] 
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=train_augmentations))
    

class EarlyStoppingHook(HookBase):
    def __init__(self, trainer, num_train_steps, num_val_steps, max_iter):
        self.trainer = trainer
        self._period = num_train_steps
        self.num_val_steps = num_val_steps
        self._max_iter = max_iter
        self.best_model = trainer.model
        self.last_valloss = []
        self._cfg = trainer.cfg.clone()
        self._cfg.DATASETS.TRAIN = trainer.cfg.DATASETS.VAL
        self._loader = iter(build_detection_train_loader(self._cfg))

    def _do_eval(self):
        #result = self.trainer.test(self.trainer.cfg, self.trainer.model)['bbox']['AP']
        total_loss = 0
        for _ in tqdm(range(self.num_val_steps)):
            data = next(self._loader)
            with torch.no_grad():
                loss_dict = self.trainer.model(data)
                total_loss += sum(loss_dict.values())
        total_loss = total_loss/self.num_val_steps
        total_loss = total_loss.cpu().detach().numpy()

        if not np.isnan(total_loss):  
            if self.last_valloss:
                if total_loss > max(self.last_valloss):
                    self.best_model = trainer.model 
                    print('New best model!')
            if len(self.last_valloss)>=patience:
                if np.argmax(self.last_valloss)==0 and total_loss < self.last_valloss[0]:
                    raise Exception("Stopping early")
                self.last_valloss[:patience-1] = self.last_valloss[1:]  
                self.last_valloss[-1] = total_loss
            else:
                self.last_valloss.append(total_loss)
        print(self.last_valloss)
        

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self._period == 0 or next_iter >= self._max_iter:
            self._do_eval()

trainer = MyTrainer(cfg)
trainer.register_hooks([EarlyStoppingHook(trainer, 
                                          num_train_steps=num_train_steps, 
                                          num_val_steps=num_val_steps, 
                                          max_iter=cfg.SOLVER.MAX_ITER)])
trainer.resume_or_load(resume=False)
device = torch.device('cuda:0')
trainer.model.to(device)
trainer.train()