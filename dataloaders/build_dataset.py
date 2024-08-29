import sys

sys.path.append("../")
from typing import Dict
from torch.utils.data import DataLoader
from monai.data import DataLoader


#############################################################################################
def build_dataset(dataset_type: str, dataset_args: Dict, augmentation_args: Dict):
    if dataset_type == "isic_torchvision":
        from .isic_dataset import ISICDataset
        from augmentations.segmentation_augmentations_tv import build_augmentations

        dataset = ISICDataset(
            data=dataset_args["data_path"],
            image_size=dataset_args["image_size"],
            input_transforms=build_augmentations(
                dataset_args["train"],
                augmentation_args,
            ),
            target_transforms=True,
        )
        return dataset
    elif dataset_type == "isic_albumentation_v2":
        from .isic_dataset import ISICDatasetA
        from augmentations.segmentation_augmentations import build_augmentations_v2

        dataset = ISICDatasetA(
            data=dataset_args["data_path"],
            transforms=build_augmentations_v2(dataset_args["train"], augmentation_args),
        )
        return dataset
    else:
        raise NotImplementedError("datasets are supported")


#############################################################################################
def build_dataloader(
    dataset,
    dataloader_args: Dict,
    config: Dict = None,
    train: bool = True,
) -> DataLoader:
    """builds the dataloader for given dataset

    Args:
        dataset (_type_): _description_
        dataloader_args (Dict): _description_
        config (Dict, optional): _description_. Defaults to None.
        train (bool, optional): _description_. Defaults to True.

    Returns:
        DataLoader: _description_
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=dataloader_args["batch_size"],
        shuffle=dataloader_args["shuffle"],
        num_workers=dataloader_args["num_workers"],
        drop_last=dataloader_args["drop_last"],
        pin_memory=dataloader_args["pin_memory"],
    )
    return dataloader


#############################################################################################
