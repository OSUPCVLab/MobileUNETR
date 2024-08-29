import torch
import monai
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F


#####################################################################
class CrossEntropyLoss(nn.Module):
    def __init__(self, loss_args=None):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction="mean")

    def __call__(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss


#####################################################################
class BinaryCrossEntropyWithLogits:
    def __init__(self, loss_args=None):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction="mean")

    def __call__(self, predictions, targets):
        loss = self._loss(predictions, targets)
        return loss


#####################################################################
class MSELoss:
    def __init__(self, loss_args=None):
        super().__init__()
        self._loss = nn.MSELoss()

    def __call__(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


#####################################################################
class L1Loss:
    def __init__(self, loss_args=None):
        super().__init__()
        self._loss = nn.L1Loss()

    def __call__(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss


#####################################################################
class DistillationLoss(nn.Module):
    # TODO: loss function not 100% verified
    def __init__(self, loss_args: Dict):
        """_summary_

        Args:
            loss_args (Dict): _description_
        """
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction="batchmean")
        self.temperature = loss_args["temperature"]
        self.lambda_param = loss_args["lambda"]
        self.dice = monai.losses.DiceLoss(
            to_onehot_y=False,
            sigmoid=True,
        )

    def forward(
        self,
        teacher_predictions: torch.Tensor,
        student_predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            teacher_predictions (torch.Tensor): _description_
            student_predictions (torch.Tensor): _description_
            targets (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        # compute probabilties
        soft_teacher = F.softmax(
            teacher_predictions.view(-1) / self.temperature,
            dim=-1,
        )
        soft_student = F.log_softmax(
            student_predictions.view(-1) / self.temperature,
            dim=-1,
        )

        # compute kl div loss
        distillation_loss = self.kl_div(soft_student, soft_teacher) * (
            self.temperature**2
        )

        # compute dice loss on student
        dice_loss = self.dice(student_predictions, targets)

        # combine via lambda
        loss = (1.0 - self.lambda_param) * (
            dice_loss + self.lambda_param * distillation_loss
        )

        return loss


class DiceBCELoss(nn.Module):
    def __init__(self, loss_args=None):
        super().__init__()
        self.dice_loss = monai.losses.DiceLoss(sigmoid=True, to_onehot_y=False)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def __call__(self, predicted, target):
        dice_loss = self.dice_loss(predicted, target) * 1
        bce_loss = self.bce_loss(predicted, target.float()) * 1
        final = dice_loss + bce_loss
        return final


#####################################################################
def build_loss_fn(loss_type: str, loss_args: Dict = None):
    if loss_type == "crossentropy":
        return CrossEntropyLoss()

    elif loss_type == "binarycrossentropy":
        return BinaryCrossEntropyWithLogits()

    elif loss_type == "MSE":
        return MSELoss()

    elif loss_type == "L1":
        return L1Loss()

    elif loss_type == "dice":
        return monai.losses.DiceLoss(to_onehot_y=False, sigmoid=True)

    elif loss_type == "dicebce":
        return DiceBCELoss()

    elif loss_type == "dicece":
        return monai.losses.DiceCELoss(to_onehot_y=True, softmax=True)
    else:
        raise ValueError("must be cross entropy or soft dice loss for now!")
