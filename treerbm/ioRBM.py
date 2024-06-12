from typing import Tuple, Union
import numpy as np
import torch
import h5py

Tensor = torch.Tensor

def get_params(filename : str, stamp : Union[str, int], device : torch.device="cpu") -> Tuple[Tensor, Tensor, Tensor]:
    """Returns the parameters of the model at the selected time stamp.

    Args:
        filename (str): filename of the model.
        stamp (Union[str, int]): Update number.
        device (torch.device): device.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Parameters of the model (vbias, hbias, weigth_matrix).
    """
    stamp = str(stamp)
    f = h5py.File(filename, "r")
    for k in f.keys():
        if "update_" in k:
            base_key = "update"
            break
        elif "epoch_" in k:
            base_key = "epoch"
            break
    key = f"{base_key}_{stamp}"
    vbias = torch.tensor(f[key]["vbias"][()], device=device)
    hbias = torch.tensor(f[key]["hbias"][()], device=device)
    weight_matrix = torch.tensor(f[key]["weight_matrix"][()], device=device)
    return (vbias, hbias, weight_matrix)

def get_epochs(filename : str) -> np.ndarray:
    """Returns the epochs at which the model has been saved.

    Args:
        filename (str): filename of the model.
        stamp (Union[str, int]): Update number.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Parameters of the model (vbias, hbias, weigth_matrix).
    """
    f = h5py.File(filename, 'r')
    alltime = []
    for key in f.keys():
        if "update" in key:
            alltime.append(int(key.replace("update_", "")))
        elif "epoch" in key:
            alltime.append(int(key.replace("epoch_", "")))
    f.close()
    # Sort the results
    alltime = np.sort(alltime)
    return alltime
