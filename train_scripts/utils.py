import os
import cv2
import wandb
import torch
import random
import numpy as np

"""
Utils File Used for Training/Validation/Testing
"""


##################################################################################################
def log_metrics(**kwargs) -> None:
    # data to be logged
    log_data = {}
    log_data.update(kwargs)

    # log the data
    wandb.log(log_data)


##################################################################################################
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar") -> None:
    # print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


##################################################################################################
def load_checkpoint(config, model, optimizer, load_optimizer=True):
    print("=> Loading checkpoint")
    checkpoint = torch.load(config.checkpoint_file_name, map_location=config.device)
    model.load_state_dict(checkpoint["state_dict"])

    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.learning_rate

    return model, optimizer


##################################################################################################
def seed_everything(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


##################################################################################################
def initialize_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight.data, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


##################################################################################################
