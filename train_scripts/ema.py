import copy
import torch.nn as nn


###############################################################################
class EMA:
    "https://github.com/scott-yjyang/DiffMIC/blob/main/ema.py"

    def __init__(
        self,
        model: nn.Module,
        mu: float = 0.999,
    ) -> None:
        self.mu = mu
        self.ema_model = {}
        self.register(model)
        self.model_copy = copy.deepcopy(model)

    def register(self, module) -> None:
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.ema_model[name] = param.data.clone()

    def update(self, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.ema_model[name].data = (
                    1.0 - self.mu
                ) * param.data + self.mu * self.ema_model[name].data

    def ema(self, module: nn.Module) -> None:
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.ema_model[name].data)

    def ema_copy(self, module: nn.Module):
        """
        Returns the model with the ema weights inserted.
        Args:
            module (nn.Module): _description_

        Returns:
            _type_: _description_
        """
        # module_copy = type(module)(module.config).to(module.config.device)
        module_copy = copy.deepcopy(module)
        # module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)

        return module_copy

    def state_dict(self):
        """
        Returns ema model
        Returns:
            _type_: _description_
        """
        return self.ema_model

    def load_state_dict(self, state_dict) -> None:
        self.ema_model = state_dict


###############################################################################
