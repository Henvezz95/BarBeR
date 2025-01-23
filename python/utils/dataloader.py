from pycocotools.coco import COCO
from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
from utils.boundary_loss_utils import calc_dist_map_chw

class BarcodeDataset(Dataset):
    def __init__(self, annotation_path, image_path, max_size=640, 
                 num_classes = 2, use_ram = True, downscale_factor=4, 
                 remove_first_ch = False, gray_scale = True, clahe=False, dist_map=True):
        self.root = image_path
        self.coco = COCO(annotation_path)
        self.ids = list(self.coco.imgs.keys())
        self.num_classes = num_classes
        self.max_size = max_size
        self.dictionary = {}
        self.use_ram = use_ram
        self.gray_scale = gray_scale
        self.clahe = clahe
        self.remove_first_ch = remove_first_ch
        self.dwf = int(downscale_factor)
        self.dist_map = dist_map
        self.transform = lambda x : np.array(transforms.ColorJitter(brightness=(0.4, 1.6), 
                                                contrast=0, 
                                                saturation=(0.3, 1.7), 
                                                hue=0)(Image.fromarray(x))).astype('float32')

    def __getitem__(self, index):  # sourcery skip: low-code-quality
        if index in self.dictionary and self.use_ram:
            img, mask = self.dictionary[index]
        else:
            coco = self.coco
            img_id = self.ids[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)

            path = coco.loadImgs(img_id)[0]['file_name']
            img = cv2.imread(os.path.join(self.root, path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W , _ = img.shape
            if W > H:
                W_new = self.max_size
                H_new = int(np.round((H*W_new)/W))
            else:
                H_new = self.max_size
                W_new = int(np.round((W*H_new)/H))

            mask = np.zeros((H,W,self.num_classes+1)).astype('float32')
            points = [np.array(target[i]['segmentation']).reshape(-1,2) for i in range(len(target))]
            mask[:,:,0] = cv2.fillPoly(np.zeros_like(img[:,:,0]), pts=points, color=[1])
            for k in range(self.num_classes):
                points = [np.array(target[i]['segmentation']).reshape(-1,2) for i in range(len(target)) if target[i]['category_id'] == k+1]
                mask[:,:,k+1] =cv2.fillPoly(np.zeros_like(img[:,:,0]), pts=points, color=[1])

            img = cv2.resize(img, (W_new, H_new), cv2.INTER_CUBIC)
            img = np.pad(img, (((self.max_size-H_new)//2, self.max_size-H_new-(self.max_size-H_new)//2),
                            ((self.max_size-W_new)//2, self.max_size-W_new-(self.max_size-W_new)//2), (0,0)))
            mask = cv2.resize(mask, (W_new, H_new), cv2.INTER_CUBIC)
            mask = np.pad(mask, (((self.max_size-H_new)//2, self.max_size-H_new-(self.max_size-H_new)//2),
                            ((self.max_size-W_new)//2, self.max_size-W_new-(self.max_size-W_new)//2),(0,0))).astype('uint8')
            mask = mask[::self.dwf, ::self.dwf]
            if self.use_ram:
                self.dictionary[index] = [img, mask]
        if np.random.uniform(0,1)>0.5:
            img = img[::-1,:,:]
            mask = mask[::-1,:,:]
        if np.random.uniform(0,1)>0.5:
            img = img[:,::-1,:]
            mask = mask[:,::-1,:]
        img = self.transform(img)
        if self.gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if self.clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
                img = clahe.apply(np.uint8(img))
                img = np.float32(img)
            img = img[:,:,np.newaxis]

        img = np.transpose(img,(2,0,1)).astype('float32')/255.0
        mask = np.transpose(mask, (2,0,1)).astype('float32')
        if self.dist_map:
            distance_map = calc_dist_map_chw(mask[1:,:,:].astype('uint8'))

        if self.dist_map:
            return (img, mask[1:,:,:], distance_map)
        return (img, mask[1:,:,:]) if self.remove_first_ch else (img, mask) 
    
    def __len__(self):
        return len(self.ids)