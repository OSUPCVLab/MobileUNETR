import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from typing import Dict


#######################################################################################
def build_augmentations(train: bool = True, augmentation_args: Dict = None):

    mean = augmentation_args["mean"]
    std = augmentation_args["std"]
    image_size = augmentation_args["image_size"]

    if train:
        train_transform = A.Compose(
            [
                A.Resize(image_size[0], image_size[1], interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.5,
                    contrast_limit=0.5,
                    p=0.6,
                ),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomResizedCrop(
                    image_size[0],
                    image_size[1],
                    interpolation=cv2.INTER_CUBIC,
                    p=0.5,
                ),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )

        return train_transform
    else:
        test_transform = A.Compose(
            [
                A.Resize(
                    image_size[0],
                    image_size[1],
                    interpolation=cv2.INTER_CUBIC,
                ),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
        return test_transform


#######################################################################################
def build_augmentations_v2(train: bool = True, augmentation_args: Dict = None):

    mean = augmentation_args["mean"]
    std = augmentation_args["std"]
    image_size = augmentation_args["image_size"]

    if train:
        train_transform = A.Compose(
            [
                A.Resize(image_size[0], image_size[1], interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.5,
                    contrast_limit=0.5,
                    p=0.6,
                ),
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                A.RandomRotate90(p=0.5),
                A.OneOf(
                    [
                        A.RandomResizedCrop(
                            image_size[0],
                            image_size[1],
                            interpolation=cv2.INTER_CUBIC,
                            p=1,
                        ),
                        A.Compose(
                            [
                                A.CropNonEmptyMaskIfExists(128, 128, p=1),
                                A.Resize(
                                    image_size[0],
                                    image_size[1],
                                    interpolation=cv2.INTER_CUBIC,
                                ),
                            ]
                        ),
                        A.Compose(
                            [
                                A.CropNonEmptyMaskIfExists(64, 64, p=1),
                                A.Resize(
                                    image_size[0],
                                    image_size[1],
                                    interpolation=cv2.INTER_CUBIC,
                                ),
                            ]
                        ),
                        A.Compose(
                            [
                                A.CropNonEmptyMaskIfExists(256, 256, p=1),
                                A.Resize(
                                    image_size[0],
                                    image_size[1],
                                    interpolation=cv2.INTER_CUBIC,
                                ),
                            ]
                        ),
                        A.Compose(
                            [
                                A.CropNonEmptyMaskIfExists(160, 160, p=1),
                                A.Resize(
                                    image_size[0],
                                    image_size[1],
                                    interpolation=cv2.INTER_CUBIC,
                                ),
                            ]
                        ),
                    ],
                    p=0.5,
                ),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )

        return train_transform
    else:
        test_transform = A.Compose(
            [
                A.Resize(image_size[0], image_size[1], interpolation=cv2.INTER_CUBIC),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.Normalize(mean, std),
                ToTensorV2(),
            ]
        )
        return test_transform


#######################################################################################
