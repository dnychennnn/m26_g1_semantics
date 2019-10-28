import os
import numpy as np
import torch
from PIL import Image
from utils import visualize



class SugarBeetDataset(object):

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images/rgb"))))


    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images/rgb/", self.imgs[idx])
        mask_path = os.path.join(self.root, "annotations/PNG/pixelwise/iMap", "{0}_GroundTruth_iMap.png".format(self.imgs[idx].split(".")[0]))
    
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.array([0,1, 10000, 20001])
        # first id is the background, so remove it 
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
       
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
       
        # convert everything into a torch.Tensor
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.float32)

        image_id = torch.tensor([idx])
        
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)