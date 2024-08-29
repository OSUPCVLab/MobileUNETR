import torch
from typing import Tuple, Dict
from fvcore.nn import FlopCountAnalysis


##########################################################################################
def flop_count_analysis(
    input_dim: Tuple,
    model: torch.nn.Module,
) -> Dict:
    """_summary_

    Args:
        input_dim (Tuple): shape: (batchsize=1, C, H, W, D(optional))
        model (torch.nn.Module): _description_
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input_tensor = torch.ones(()).new_empty(
        (1, *input_dim),
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    flops = FlopCountAnalysis(model, input_tensor)
    model_flops = flops.total()
    print(f"Total trainable parameters: {round(trainable_params * 1e-6, 2)} M")
    print(f"MAdds: {round(model_flops * 1e-9, 2)} G")

    out = {
        "params": round(trainable_params * 1e-6, 2),
        "flops": round(model_flops * 1e-9, 2),
    }

    return out
