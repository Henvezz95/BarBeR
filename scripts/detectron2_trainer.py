import torch
import os
import yaml
import json
import getopt
import sys
from tqdm import tqdm
import numpy as np

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import HookBase

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def parse_inputs(file_path, argv):
    config_path = None
    output_path = None
    file_name = file_path.split('/')[-1]
    try:
        opts, _ = getopt.getopt(argv, "hc:o:", ["cfile=", "opath="])
    except getopt.GetoptError:
        print(file_name, '-c <configfile> -o <trained_model_output_path>')
        print('The configuration file must be in yaml format, the output path does not need an extension, it will be added automatically (.pt)')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(file_name, '-c <configfile> -o <trained_model_output_path>')
            print('The configuration file must be in yaml format, the output path does not need an extension, it will be added automatically (.pt)')
            sys.exit()
        elif opt in ("-c", "--cfile"):
            config_path = arg
        elif opt in ("-o", "--opath"):
            output_path = arg

    if config_path == None:
        print(file_name, '-c <configfile> -o <trained_model_output_path>')
        print('The configuration file must be in yaml format, the output path does not need an extension, it will be added automatically (.pt)')
        sys.exit(2)
    if output_path == None:
        print(file_name, '-c <configfile> -o <trained_model_output_path>')
        print('The configuration file must be in yaml format, the output path does not need an extension, it will be added automatically (.pt)')
        sys.exit(2)

    return config_path, output_path+'.pt'


if __name__ == "__main__":
    config_path, output_path = parse_inputs(sys.argv[0], sys.argv[1:])
    
    # Read YAML file
    with open(config_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    annotations_path = data_loaded['coco_annotations_path']
    images_path = data_loaded['images_path']

    with open(f'{annotations_path}datasets_info.json') as json_file:
        datasets_info = json.load(json_file)
    num_categories = len(datasets_info['categories'])

    register_coco_instances("my_dataset_train", {}, f'{annotations_path}train.json', images_path)
    register_coco_instances("my_dataset_val", {}, f'{annotations_path}val.json', images_path)
    train_metadata = MetadataCatalog.get("my_dataset_train")
    train_dataset_dicts = DatasetCatalog.get("my_dataset_train")
    val_metadata = MetadataCatalog.get("my_dataset_val")
    val_dataset_dicts = DatasetCatalog.get("my_dataset_val")
    num_train_samples=len(train_dataset_dicts)
    num_val_samples=len(val_dataset_dicts)

    if 'pretrained_model' in data_loaded:
        pre_trained_model = data_loaded['pretrained_model']
    else:
        pre_trained_model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

    if 'batch' in data_loaded:
        batch_size =  data_loaded['batch']
    else:   
        batch_size = 4

    if 'patience' in data_loaded:
        patience = data_loaded['patience']
    else:
        patience = 20

    if 'epochs' in data_loaded:
        epochs = data_loaded['epochs']
    else:
        epochs = 200

    if 'fliplr' in data_loaded:
        fliplr = data_loaded['fliplr']
    else:
        fliplr = 0.5

    if 'flipud' in data_loaded:
        flipud = data_loaded['flipud']
    else:
        flipud = 0.5

    if 'roi_heads' in data_loaded:
        roi_heads = data_loaded['roi_heads']
    else:
        roi_heads = 512

    if 'learning_rate' in data_loaded:
        lr = data_loaded['learning_rate']
    else:
        lr = 1e-3

    imgsz = data_loaded['imgsz']

    num_train_steps = num_train_samples//batch_size
    num_val_steps = num_val_samples//batch_size

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(pre_trained_model))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pre_trained_model)

    cfg.DATASETS.TRAIN = ("my_dataset_train")
    cfg.DATASETS.VAL = ("my_dataset_val",)

    cfg.INPUT.MIN_SIZE = imgsz
    cfg.INPUT.MIN_SIZE_TRAIN = imgsz
    cfg.INPUT.MIN_SIZE_TEST = imgsz
    cfg.INPUT.MAX_SIZE = imgsz
    cfg.INPUT.MAX_SIZE_TRAIN = imgsz
    cfg.INPUT.MAX_SIZE_TEST = imgsz
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = epochs*num_train_steps   
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_heads 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_categories+1 

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    class MyTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, cfg, True, output_folder)
        
        @classmethod
        def build_train_loader(cls, cfg):
            train_augmentations = [
                T.ResizeShortestEdge(short_edge_length=[imgsz,imgsz], max_size=imgsz),
                T.RandomBrightness(0.4, 1.6),
                T.RandomSaturation(0.3, 1.7),
                T.RandomFlip(prob=fliplr, horizontal=True, vertical=False),
                T.RandomFlip(prob=flipud, horizontal=False, vertical=True),
            ] 
            return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=train_augmentations))
        

    class EarlyStoppingHook(HookBase):
        def __init__(self, trainer, num_train_steps, num_val_steps, max_iter):
            self.trainer = trainer
            self.current_epoch = 0
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
            print([x.item for x in self.last_valloss])
            

        def after_step(self):
            if self.current_epoch == 0:
                self.current_epoch+=1
                print('Epoch: ', self.current_epoch, '/', epochs)
            next_iter = self.trainer.iter + 1
            if next_iter % self._period == 0 or next_iter >= self._max_iter:
                self._do_eval()
                self.current_epoch+=1
                print('Epoch: ', self.current_epoch, '/', epochs)

    trainer = MyTrainer(cfg)
    trainer.register_hooks([EarlyStoppingHook(trainer, 
                                            num_train_steps=num_train_steps, 
                                            num_val_steps=num_val_steps, 
                                            max_iter=cfg.SOLVER.MAX_ITER)])
    trainer.resume_or_load(resume=False)
    device = torch.device('cuda:0')
    trainer.model.to(device)
    
    try:
        trainer.train()
    except:
        print('Saving best model')

    torch.save(trainer._hooks[-1].best_model, output_path)