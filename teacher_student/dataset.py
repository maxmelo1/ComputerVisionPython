import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import functional as TF

class VOCDataset(Dataset):
    def __init__(self, base_dir, list_path, crop_size, transform=None, n_crops=1):
        self.img_dir = os.path.join(base_dir, 'JPEGImages')
        self.mask_dir = os.path.join(base_dir, 'SegmentationClass')
        self.list_path = list_path
        self.crop_size = crop_size
        self.transform = transform
        self.n_crops = n_crops
        self.base_dir = base_dir

        self.img_ids = [img_id.strip().split(' ')[0].lstrip('/') for img_id in open(self.list_path).readlines()]
        self.mask_ids = [img_id.strip().split(' ')[1].lstrip('/') for img_id in open(self.list_path).readlines()]


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.img_ids[idx])
        img = Image.open(img_path).convert('RGB')

        mask_path = os.path.join(self.base_dir, self.mask_ids[idx])
        mask = Image.open(mask_path)

        if self.transform is not None:
            img, mask = self.transform(img, mask)  

        return img, mask



def denormalize(img):
    img = img.clone()
    img[:, 0] = img[:, 0] * 0.229 + 0.485
    img[:, 1] = img[:, 1] * 0.224 + 0.456
    img[:, 2] = img[:, 2] * 0.225 + 0.406
    return img


def random_crops_from_tensors(img_t: torch.Tensor,
                              mask_t: torch.Tensor,
                              crop_size: int,
                              n_crops: int):
    """
    Args:
      img_t: (C, H, W) image tensor
      mask_t: (H, W) or (1, H, W) label tensor (index map)
      crop_size: int, square crop size
      n_crops: number of random crops

    Returns:
      crop_imgs: (n_crops, C, crop_size, crop_size)
      crop_masks: (n_crops, crop_size, crop_size)
      bboxes: (n_crops, 4) [x1, y1, x2, y2] in input tensor coords
    """
    assert img_t.dim() == 3, "img_t must be (C,H,W)"
    if mask_t.dim() == 3 and mask_t.size(0) == 1:
        mask_t = mask_t[0]
    assert mask_t.dim() == 2, "mask_t must be (H,W) or (1,H,W)"

    C, H, W = img_t.shape
    th = tw = crop_size

    # pad if needed to reach min crop size
    pad_h = max(0, th - H)
    pad_w = max(0, tw - W)
    if pad_h > 0 or pad_w > 0:
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h), value=0.0)
        mask_t = F.pad(mask_t.unsqueeze(0), (0, pad_w, 0, pad_h), value=255).squeeze(0)
        H += pad_h
        W += pad_w

    crop_imgs = []
    crop_masks = []
    bboxes = []

    for _ in range(max(n_crops, 0)):
        x1 = random.randint(0, W - tw)
        y1 = random.randint(0, H - th)
        x2 = x1 + tw
        y2 = y1 + th

        crop_imgs.append(img_t[:, y1:y2, x1:x2])
        crop_masks.append(mask_t[y1:y2, x1:x2])
        bboxes.append([x1, y1, x2, y2])

    if len(crop_imgs) == 0:
        return (torch.empty(0, C, th, tw, dtype=img_t.dtype, device=img_t.device),
                torch.empty(0, th, tw, dtype=mask_t.dtype, device=img_t.device),
                torch.empty(0, 4, dtype=torch.long, device=img_t.device))

    return torch.stack(crop_imgs, 0), torch.stack(crop_masks, 0), torch.tensor(bboxes, dtype=torch.long, device=img_t.device)