import torch
import numpy as np
import torchvision
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as tvF


##########################################################################################################
class ISICDataset(torch.utils.data.Dataset):
    def __init__(self, data, image_size, input_transforms, target_transforms):
        """
        Initialize Class
        Args:
            config (_type_): _description_
            data (_type_): _description_
            input_transforms (_type_): _description_
            target_transforms (_type_): _description_
        """
        self.data = pd.read_csv(data)
        self.image_size = image_size  # tuple
        self.input_transforms = input_transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get the associated_image
        image_id = self.data.iloc[index]

        # get the image
        image_name = image_id["image"]
        image = Image.open(image_name).convert("RGB")

        # get the mask path
        mask_id = image_id["mask"]
        mask = Image.open(mask_id)

        # transform input image
        if self.input_transforms:
            image = self.input_transforms(image)

        # transform target image
        if self.target_transforms:
            mask = tvF.resize(
                mask,
                self.image_size,
                torchvision.transforms.InterpolationMode.NEAREST,
            )
            mask = np.array(mask).astype(int)
            mask[mask == 255] = 1
            mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

        out = {
            "image": image,  # [3, H, W]
            "mask": mask,  # [1, H, W] 1 b/c binary mask
        }

        return out


##########################################################################################################
class ISICDatasetA(torch.utils.data.Dataset):
    """
    Dataset used for albumentation augmentations
    """

    def __init__(self, data, transforms):
        """
        Initialize Class
        Args:
            config (_type_): _description_
            data (_type_): _description_
            input_transforms (_type_): _description_
            target_transforms (_type_): _description_
        """
        self.data = pd.read_csv(data)
        # self.data = pd.concat([self.data] * 2).sample(frac=1).reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get the associated_image
        image_id = self.data.iloc[index]

        # get the image
        image_name = image_id["image"]
        image = Image.open(image_name).convert("RGB")
        image = np.array(image, dtype=np.uint8)

        # get the mask path
        mask_id = image_id["mask"]
        mask = Image.open(mask_id)
        mask = np.array(mask, dtype=np.uint8)
        mask[mask == 255] = 1

        # transform
        if self.transforms:
            out = self.transforms(image=image, mask=mask)
            image = out["image"]
            mask = out["mask"]

        out = {
            "image": image,  # [3, H, W]
            "mask": mask.unsqueeze(0),  # [1, H, W] 1 b/c binary mask
        }

        return out
