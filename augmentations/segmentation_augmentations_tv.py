import torchvision.transforms as transforms
from typing import Dict


# [0.7128, 0.6000, 0.5532], [0.1577, 0.1662, 0.1829]
#######################################################################################
def build_augmentations(train: bool = True, augmentation_args: Dict = None):
    mean = augmentation_args["mean"]
    std = augmentation_args["std"]
    image_size = augmentation_args["image_size"]

    if train:
        train_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size[0], image_size[1]),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.autoaugment.TrivialAugmentWide(
                    interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        return train_transform
    else:
        test_transform = transforms.Compose(
            [
                # transforms.RandomRotation(180),
                transforms.Resize(
                    (image_size[0], image_size[1]),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        return test_transform
