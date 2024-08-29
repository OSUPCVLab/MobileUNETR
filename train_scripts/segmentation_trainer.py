import os
import torch
from tqdm import tqdm
from typing import Dict
from termcolor import colored
from torch.utils.data import DataLoader
import torch.nn.functional as F
from monai.metrics import MeanIoU, DiceMetric
from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Optimizer
from .ema import EMA


######################################################################################
class Segmentation_Trainer:
    def __init__(
        self,
        config: Dict,
        model: torch.nn.Module,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        warmup_scheduler: LRScheduler,
        training_scheduler: LRScheduler,
        accelerator=None,
    ) -> None:
        """
        Trainer Base Class

        Args:
            config (Dict): _description_
            model (torch.nn.Module): _description_
            optimizer (torch.optim.Optimizer): _description_
            criterion (torch.nn.Module): _description_
            train_dataloader (DataLoader): _description_
            val_dataloader (DataLoader): _description_
            warmup_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            training_scheduler (torch.optim.lr_scheduler.LRScheduler): _description_
            accelerator (_type_, optional): _description_. Defaults to None.
        """
        # config
        self.config = config
        if accelerator.is_main_process:
            self._configure_trainer()

        # model components
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        # training scheduler
        self.warmup_scheduler = warmup_scheduler
        self.training_scheduler = training_scheduler
        self.scheduler = None

        # accelerate object
        self.accelerator = accelerator

        # create ema model
        self.ema_model = None
        if self.accelerator.is_main_process and self.ema_enabled:
            self.ema_model = self._create_ema_model(gpu_id=None)

        # get wandb object
        self.wandb_tracker = self.accelerator.get_tracker("wandb")

        # track train and val loss values
        self.current_epoch = 0
        self.epoch_train_loss = 0.0
        self.best_train_loss = 100.0
        self.epoch_val_loss = 0.0
        self.best_val_loss = 100.0

        # external metric functions we can add
        self.iou_metric = None
        self.dice_metric = None
        if self.use_miou:
            miou_args = self.config["metrics"]["mean_iou"]["mean_iou_args"]
            self.iou_metric = MeanIoU(**miou_args)
        if self.use_dice:
            dice_args = self.config["metrics"]["dice"]["dice_args"]
            self.dice_metric = DiceMetric(**dice_args)

        # track performance metrics
        self.epoch_val_iou = 0.0
        self.best_val_iou = 0.0
        self.epoch_val_dice = 0.0
        self.best_val_dice = 0.0
        self.ema_val_iou = 0.0

        # misc params
        self.eval_ema = False  # var that indicates if ema model was used to validate
        self.warming_up = True  # var that indicates if we are in a warming up stage
        self.val_ema_model = None

        print(self.iou_metric, self.dice_metric, self.metrics_type)

    def _configure_trainer(self) -> None:
        """
        Configures useful config variables
        """
        self.num_epochs = self.config["training_parameters"]["num_epochs"]
        self.print_every = self.config["training_parameters"]["print_every"]
        self.ema_enabled = self.config["ema"]["enabled"]
        self.print_ema_every = self.config["ema"]["print_ema_every"]
        self.warmup_enabled = self.config["warmup_scheduler"]["enabled"]
        self.warmup_epochs = self.config["warmup_scheduler"]["warmup_epochs"]
        self.cutoff_epoch = self.config["training_parameters"]["cutoff_epoch"]
        self.calculate_metrics = self.config["training_parameters"]["calculate_metrics"]
        self.checkpoint_save_dir = self.config["training_parameters"][
            "checkpoint_save_dir"
        ]
        self.use_miou = self.config["metrics"]["mean_iou"]["enabled"]
        self.use_dice = self.config["metrics"]["dice"]["enabled"]
        self.metrics_type = self.config["metrics"]["type"]
        self.num_classes = self.config["variables"]["num_classes"]

    def _load_checkpoint(self):
        raise NotImplementedError

    def _create_ema_model(self, gpu_id: int) -> torch.nn.Module:
        self.accelerator.print(f"[info] -- creating ema model")
        ema_model = EMA(model=self.model)
        return ema_model

    def _train_step(self) -> None:
        """
        Runs a Single Training Epoch
        """
        # Initialize the training loss for the current epoch
        epoch_avg_loss = 0.0

        # set model to train
        self.model.train()

        # run training
        progress_bar = tqdm(
            desc=f"Epoch: {self.current_epoch} -- Training",
            total=len(self.train_dataloader),
            leave=True,
            bar_format="{l_bar}{bar}{r_bar}",
            ncols=100,
            colour="MAGENTA",
        )
        for index, data in enumerate(self.train_dataloader):
            with self.accelerator.accumulate(self.model):
                # get data ex: (data, target)
                images = data["image"]
                mask = data["mask"]

                # zero out existing gradients
                self.optimizer.zero_grad()

                # forward pass
                predicted = self.model.forward(images)

                # calculate loss
                loss = self.criterion(predicted, mask)

                # backward pass
                self.accelerator.backward(loss)

                # update gradients
                self.optimizer.step()

                # model update with ema if available
                if (
                    self.ema_enabled
                    and (self.accelerator.is_main_process)
                    # and (self.warming_up == False)
                ):
                    self.ema_model.update(self.model)

                # update loss
                epoch_avg_loss += loss.item()

                # update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix(
                    tr_loss=f"{(epoch_avg_loss / (index + 1)):.5f}",
                )
                progress_bar.refresh()

            # break

        # calculate and update epoch average loss
        self.epoch_train_loss = epoch_avg_loss / (index + 1)

    def _val_step(self, use_ema: bool = False) -> None:
        """
        Runs a Single Validation Epoch

        Args:
            use_ema (bool, optional): _description_. Defaults to False.
        """
        # Initialize the training loss for the current Epoch
        epoch_avg_loss = 0.0
        total_iou = 0.0

        # set model to eval mode
        self.model.eval()
        if use_ema:
            self.val_ema_model = self.ema_model.ema_copy(self.model)
            self.val_ema_model.eval()

        # run validation
        progress_bar = tqdm(
            desc=f"Epoch: {self.current_epoch} -- Validation",
            total=len(self.val_dataloader),
            leave=True,
            bar_format="{l_bar}{bar}{r_bar}",
            ncols=100,
            colour="YELLOW",
        )
        with torch.no_grad():
            for index, (data) in enumerate(self.val_dataloader):
                # get data ex: (data, target)
                images = data["image"]
                mask = data["mask"]

                # forward pass
                if use_ema:
                    self.eval_ema = True
                    predicted = self.val_ema_model.forward(images)
                else:
                    self.eval_ema = False
                    predicted = self.model.forward(images)

                # calculate loss
                loss = self.criterion(predicted, mask)

                if self.calculate_metrics:
                    if self.use_miou and self.metrics_type == "binary":
                        self._calc_binary_meaniou_metric(predicted, mask)
                    if self.use_miou and self.metrics_type == "multiclass":
                        self._calc_multiclass_meaniou_metric(predicted, mask)
                    if self.use_dice and self.metrics_type == "binary":
                        self._calc_binary_dice_metric(predicted, mask)
                    if self.use_dice and self.metrics_type == "multiclass":
                        self._calc_multiclass_dice_metric(predicted, mask)

                # update loss for the current batch
                epoch_avg_loss += loss.item()

                # update progress bar
                progress_bar.update(1)
                progress_bar.refresh()

        # if use_ema:
        #     self.epoch_val_iou = total_iou / float(index + 1)
        # else:
        self.epoch_val_loss = epoch_avg_loss / float(index + 1)
        if self.use_miou:
            self.epoch_val_iou = self.iou_metric.aggregate().item()
            self.iou_metric.reset()
        if self.use_dice:
            self.epoch_val_dice = self.dice_metric.aggregate().item()
            self.dice_metric.reset()

    def _calc_multiclass_meaniou_metric(self, predicted, labels) -> float:
        predictions, labels = self.accelerator.gather_for_metrics((predicted, labels))
        predictions = torch.softmax(predictions, dim=1)  # [B, C, H, W]
        predictions = torch.argmax(predictions, dim=1, keepdim=False)  # [B, H, W]
        labels = torch.argmax(labels, dim=1, keepdim=False)
        # predictions = np.split(predictions, 4, 0)
        # labels = np.split(labels, 4, 0)
        predictions = [predictions[i] for i in range(predictions.shape[0])]
        predictions = [
            torch.moveaxis(F.one_hot(p, num_classes=self.num_classes), 2, 0)
            for p in predictions
        ]
        labels = [labels[i] for i in range(labels.shape[0])]
        labels = [
            torch.moveaxis(F.one_hot(p, num_classes=self.num_classes), 2, 0)
            for p in labels
        ]

        assert (
            predictions[0].shape == labels[0].shape
        ), "predictions and labels have different shapes"
        self.iou_metric(y_pred=predictions, y=labels)

    def _calc_binary_meaniou_metric(self, predicted, labels) -> float:
        predictions, labels = self.accelerator.gather_for_metrics((predicted, labels))
        predictions = torch.sigmoid(predictions)  # [B, H, W]
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0

        # input to function must be [BCHW[D]]
        assert (
            predictions.shape == labels.shape
        ), "predictions and labels have different shapes"

        # predictions = torch.split(predictions, predictions.shape[0])
        # labels = torch.split(labels, labels.shape[0])

        # print(predictions[0].shape, labels[0].shape)

        self.iou_metric(y_pred=predictions, y=labels)

    def _calc_multiclass_dice_metric(self, predicted, labels) -> float:
        # TODO: Unteseted -- figure out how to add in num clases automatically
        predictions, labels = self.accelerator.gather_for_metrics((predicted, labels))
        predictions = torch.softmax(predictions, dim=1)  # [B, C, H, W]
        predictions = torch.argmax(predictions, dim=1, keepdim=False)  # [B, H, W]
        labels = torch.argmax(labels, dim=1, keepdim=False)
        # predictions = np.split(predictions, 4, 0)
        # labels = np.split(labels, 4, 0)
        predictions = [predictions[i] for i in range(predictions.shape[0])]
        predictions = [
            torch.moveaxis(F.one_hot(p, num_classes=self.num_classes), 2, 0)
            for p in predictions
        ]
        labels = [labels[i] for i in range(labels.shape[0])]
        labels = [
            torch.moveaxis(F.one_hot(p, num_classes=self.num_classes), 2, 0)
            for p in labels
        ]

        assert (
            predictions.shape == labels.shape
        ), "predictions and labels have different shapes"
        self.dice_metric(y_pred=predictions, y=labels)

    def _calc_binary_dice_metric(self, predicted, labels) -> float:
        predictions, labels = self.accelerator.gather_for_metrics((predicted, labels))
        predictions = torch.sigmoid(predictions)  # [B, H, W]
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0

        # input to function must be [BCHW[D]]
        assert (
            predictions.shape == labels.shape
        ), "predictions and labels have different shapes"

        # predictions = torch.split(predictions, predictions.shape[0])
        # labels = torch.split(labels, labels.shape[0])

        # print(predictions[0].shape, labels[0].shape)

        self.dice_metric(y_pred=predictions, y=labels)

    def _run_train_val(self) -> None:
        """
        Run Full Training and Validation.
        """
        # Tell wandb to watch the model and optimizer values
        if self.accelerator.is_main_process:
            self.wandb_tracker.run.watch(
                self.model,
                self.criterion,
                log="all",
                log_freq=10,
                log_graph=True,
            )

        # Run Complete Training and Validation
        for epoch in range(self.num_epochs):
            # update epoch
            self.current_epoch = epoch

            # update scheduler
            if self.warmup_enabled or self.current_epoch == 0:
                self._update_scheduler()

            # run a single training step
            self._train_step()

            # run a single validation step
            if self.ema_enabled and (epoch % self.print_ema_every == 0):
                self._val_step(use_ema=True)
            else:
                self._val_step(use_ema=False)

            if self.accelerator.is_main_process:
                # update metrics
                self._update_metrics()

                # log metrics
                self._log_metrics()

                # update scheduler
                self.scheduler.step()

                # save and print
                self._save_and_print()

    def _update_scheduler(self) -> None:
        """
        Updates the Learning Rate Scheduler
        """
        if self.warmup_enabled:
            self.warming_up = True
            if self.current_epoch == 0:
                self.accelerator.print(
                    colored(f"\n[info] -- warming up learning rate \n", color="red")
                )
                self.scheduler = self.warmup_scheduler
            elif self.current_epoch == self.warmup_epochs:
                self.accelerator.print(
                    colored(
                        f"\n[info] -- switching to learning rate decay schedule \n",
                        color="red",
                    )
                )
                self.scheduler = self.training_scheduler
        else:
            self.warming_up = False
            self.accelerator.print(
                colored(
                    f"\n[info] -- setting learning rate decay schedule \n",
                    color="red",
                )
            )
            self.scheduler = self.training_scheduler

    def _update_metrics(self) -> None:
        """
        Update Loss Values and Metrics After Each Epoch.
        """
        # update training loss
        if self.epoch_train_loss <= self.best_train_loss:
            self.best_train_loss = self.epoch_train_loss

        # update validation loss
        if self.epoch_val_loss <= self.best_val_loss:
            self.best_val_loss = self.epoch_val_loss

        # update iou metric
        if self.calculate_metrics:
            if self.epoch_val_iou >= self.best_val_iou:
                self.best_val_iou = self.epoch_val_iou

    def _log_metrics(self) -> None:
        """
        Log Metrics To Wandb/MLFlow etc Platform
        """
        # data to be logged
        log_data = {
            "epoch": self.current_epoch,
            "train_loss": self.epoch_train_loss,
            "val_loss": self.epoch_val_loss,
            "mean_iou": self.epoch_val_iou,
        }
        # log the data
        self.accelerator.log(log_data)

    def _save_and_print(self) -> None:
        """_summary_"""
        # TODO: Ugly Redo
        # print only on the first gpu
        saved_loss = False
        if self.epoch_val_loss <= self.best_val_loss:
            # change path name based on cutoff epoch
            if self.current_epoch <= self.cutoff_epoch:
                loss_save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_loss_state",
                )
            else:
                loss_save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_loss_state_post_cutoff",
                )
            saved_loss = True

        saved_metric = False
        if self.epoch_val_iou >= self.best_val_iou:
            # change path name based on cutoff epoch
            if self.current_epoch <= self.cutoff_epoch:
                metric_save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_iou_state",
                )
            else:
                metric_save_path = os.path.join(
                    self.checkpoint_save_dir,
                    "best_iou_state_post_cutoff",
                )
            saved_metric = True

        if saved_loss and saved_metric:
            self.accelerator.print(
                f"epoch -- {colored(str(self.current_epoch).zfill(4), color='green')} || "
                f"train loss -- {colored(f'{self.epoch_train_loss:.5f}', color='green')} || "
                f"val loss -- {colored(f'{self.epoch_val_loss:.5f}', color='green')} || "
                f"lr -- {colored(f'{self.scheduler.get_last_lr()[0]:.8f}', color='green')} || "
                f"ema -- {self.eval_ema} || "
                f"val mean_iou -- {colored(f'{self.best_val_iou:.5f}', color='green')} -- saved metric/loss \n"
            )
            self._save_checkpoint(loss_save_path)
            self._save_checkpoint(metric_save_path)
        elif saved_metric:
            self.accelerator.print(
                f"epoch -- {colored(str(self.current_epoch).zfill(4), color='green')} || "
                f"train loss -- {colored(f'{self.epoch_train_loss:.5f}', color='green')} || "
                f"val loss -- {colored(f'{self.epoch_val_loss:.5f}', color='green')} || "
                f"lr -- {colored(f'{self.scheduler.get_last_lr()[0]:.8f}', color='green')} || "
                f"ema -- {self.eval_ema} || "
                f"val mean_iou -- {colored(f'{self.best_val_iou:.5f}', color='green')} -- saved metric \n"
            )
            self._save_checkpoint(metric_save_path)
        elif saved_loss:
            self.accelerator.print(
                f"epoch -- {colored(str(self.current_epoch).zfill(4), color='green')} || "
                f"train loss -- {colored(f'{self.epoch_train_loss:.5f}', color='green')} || "
                f"val loss -- {colored(f'{self.epoch_val_loss:.5f}', color='green')} || "
                f"lr -- {colored(f'{self.scheduler.get_last_lr()[0]:.8f}', color='green')} || "
                f"ema -- {self.eval_ema} || "
                f"val mean_iou -- {colored(f'{self.best_val_iou:.5f}', color='green')} -- saved loss \n"
            )
            self._save_checkpoint(loss_save_path)
        else:
            self.accelerator.print(
                f"epoch -- {str(self.current_epoch).zfill(4)} || "
                f"train loss -- {self.epoch_train_loss:.5f} || "
                f"val loss -- {self.epoch_val_loss:.5f} || "
                f"lr -- {self.scheduler.get_last_lr()[0]:.8f} || "
                f"ema -- {self.eval_ema} || "
                f"val mean_iou -- {self.epoch_val_iou:.5f}"
            )

    def _save_checkpoint(self, filename: str) -> None:
        """_summary_

        Args:
            filename (str): _description_
        """
        # saves the ema model checkpoint if availabale
        # TODO: ema saving untested
        if self.eval_ema and self.accelerator.is_main_process:
            checkpoint = {
                "state_dict": self.ema_model.state_dict(),
            }
            torch.save(checkpoint, f"{os.path.dirname(filename)}/ema_model_ckpt.pth")
            self.val_ema_model = (
                None  # set ema model to None to avoid duplicate model saving
            )

        # standard model checkpoint
        self.accelerator.save_state(filename, safe_serialization=False)

    def train(self) -> None:
        """
        Runs a full training and validation of the dataset.
        """
        self._run_train_val()
        self.accelerator.free_memory()

    def evaluate(self, **kwargs) -> None:
        pass
