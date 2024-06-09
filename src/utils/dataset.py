import os
import torch
from PIL import Image
from typing import Tuple, List
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    """
    Creates a dataset class for the DataLoader.
    Images are assumed to have the same filename as their corresponding masks.

    Parameters
    ----------
    img_dir: str
        The parent directory containing all images.

    mask_dir: str
        The parent directory containing all masks.

    mean: float
        The mean pixel value of all images in the training set for normalization.
    
    std: float
        The pixel-level standard deviation of all images in the training set for normalization.

    Returns
    -------
    img: torch.Tensor
        The grayscale image tensor of shape (1, height, width).
        The range of pixel values for this tensor will be from 0 to 1.

    mask: torch.Tensor
        The RGB mask tensor of shape (3, height, width).
        The range of pixel values for this tensor will be from 0 to 1.

    Attributes
    ----------
    imgs: List[str]
        List of image names.
    
    masks: List[str]
        List of mask names.

    Raises
    ------
    ValueError
        If the number of images and masks do not match. 
        The error message will provide a list of images without corresponding masks and vice versa.
    """

    def __init__(self, img_dir: str, mask_dir: str, mean: float, std: float):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mean = mean
        self.std = std

        self.imgs: List[str] = os.listdir(img_dir)
        self.masks: List[str] = os.listdir(mask_dir)

        if len(self.imgs) != len(self.masks):
            self.mismatch_error()
            
    def mismatch_error(self):
        error_message = "The number of images and masks do not match.\n"
        unmatched_imgs = [img for img in self.imgs if img not in self.masks]
        unmatched_masks = [mask for mask in self.masks if mask not in self.imgs]
        
        if unmatched_imgs:
            error_message += "\nList of images without masks:\n"
            
            for img_name in unmatched_imgs:
                error_message += f"- {img_name}\n"

        if unmatched_masks:
            error_message += "\nList of masks without images:\n"

            for mask_name in unmatched_masks:
                error_message += f"- {mask_name}\n"
    
        raise ValueError(error_message)

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.imgs[idx]
        mask_name = img_name

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("RGB")

        normalize = transforms.Normalize([self.mean], [self.std])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        img, mask = preprocess(img), transforms.ToTensor()(mask)

        return img, mask
