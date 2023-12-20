import torch
from typing import Tuple, Optional
from tqdm import tqdm

Tensor = torch.Tensor

def profile_hiddens(v : Tensor, hbias : Tensor, weight_matrix : Tensor) -> Tensor:
    """Computes the hidden profile conditioned on the state of the visible layer.

    Args:
        v (Tensor): Visible layer.
        hbias (Tensor): Hidden bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tensor: Hidden magnetizations.
    """
    mh = torch.sigmoid(hbias + v @ weight_matrix)
    return mh

def profile_visibles(h : Tensor, vbias : Tensor, weight_matrix : Tensor) -> Tensor:
    """Samples the visible layer conditioned on the hidden layer.

    Args:
        h (Tensor): Hidden layer.
        vbias (Tensor): Visible bias.
        weight_matrix (Tensor): Weight matrix.

    Returns:
        Tensor: Visible units.
    """
    mv = torch.sigmoid(vbias + h @ weight_matrix.T)
    return mv

@torch.jit.script
def iterate_mf1(X : Tuple[Tensor, Tensor], params : Tuple[Tensor, Tensor, Tensor], alpha : float=1e-6, max_iter : int=2000, rho : float=0.) -> Tuple[Tensor, Tensor]:
    """Iterates the mean field self-consistency equations at first order (naive mean field), starting from the visible units X, until convergence.
    Args:
        X (Tuple[Tensor, Tensor]): Initial conditions (visible and hidden magnetizations).
        params (Tuple[Tensor, Tensor, Tensor]): Parameters of the model (vbias, hbias, weight_matrix).
        alpha (float, optional): Convergence threshold. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        rho (float, optional): Dumping parameter. Defaults to 0..
    Returns:
        Tuple[Tensor, Tensor]: Fixed points of (visible magnetizations, hidden magnetizations)
    """
    
    mv, mh = X
    iterations = 0
    vbias, hbias, weight_matrix = params
    while True:
        mv_prev = torch.clone(mv)
        mh_prev = torch.clone(mh)
        field_h = hbias + (mv @ weight_matrix)
        mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)
        field_v = vbias + (mh @ weight_matrix.mT)
        mv = rho * mv_prev + (1. - rho) * torch.sigmoid(field_v)
        eps1 = torch.abs(mv - mv_prev).max()
        eps2 = torch.abs(mh - mh_prev).max()
        if (eps1 < alpha) and (eps2 < alpha):
            break
        iterations += 1
        if iterations >= max_iter:
            break
    return (mv, mh)

@torch.jit.script
def iterate_mf2(X : Tuple[Tensor, Tensor], params : Tuple[Tensor, Tensor, Tensor], alpha : float=1e-6, max_iter : int=2000, rho : float=0.) -> Tuple[Tensor, Tensor]:
    """Iterates the mean field self-consistency equations at second order (TAP equations), starting from the visible units X, until convergence.
    Args:
        X (Tuple[Tensor, Tensor]): Initial conditions (visible and hidden magnetizations).
        params (Tuple[Tensor, Tensor, Tensor]): Parameters of the model (vbias, hbias, weight_matrix).
        alpha (float, optional): Convergence threshold. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        rho (float, optional): Dumping parameter. Defaults to 0..
    Returns:
        Tuple[Tensor, Tensor]: Fixed points of (visible magnetizations, hidden magnetizations)
    """
    mv, mh = X
    vbias, hbias, weight_matrix = params
    weight_matrix2 = torch.square(weight_matrix)
    iterations = 0

    while True:
        mv_prev = torch.clone(mv)
        mh_prev = torch.clone(mh)
        
        dmv = mv - torch.square(mv)
        
        field_h = hbias \
            + (mv @ weight_matrix) \
            + (0.5 - mh) * (dmv @ weight_matrix2)
        mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)
        
        dmh = mh - torch.square(mh)
        field_v = vbias \
            + (mh @ weight_matrix.mT) \
            + (0.5 - mv) * (dmh @ weight_matrix2.mT)
        mv = rho * mv_prev + (1. - rho) * torch.sigmoid(field_v)
        eps1 = torch.abs(mv - mv_prev).max()
        eps2 = torch.abs(mh - mh_prev).max()
        if (eps1 < alpha) and (eps2 < alpha):
            break
        iterations += 1
        if iterations >= max_iter:
            break
    return (mv, mh)

@torch.jit.script
def iterate_mf3(X : Tuple[Tensor, Tensor], params : Tuple[Tensor, Tensor, Tensor], alpha : float=1e-6, max_iter : int=2000, rho : float=0.) -> Tuple[Tensor, Tensor]:
    """Iterates the mean field self-consistency equations at third order, starting from the visible units X, until convergence.
    Args:
        X (Tuple[Tensor, Tensor]): Initial conditions (visible and hidden magnetizations).
        params (Tuple[Tensor, Tensor, Tensor]): Parameters of the model (vbias, hbias, weight_matrix).
        alpha (float, optional): Convergence threshold. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        rho (float, optional): Dumping parameter. Defaults to 0..
    Returns:
        Tuple[Tensor, Tensor]: Fixed points of (visible magnetizations, hidden magnetizations)
    """
    mv, mh = X
    vbias, hbias, weight_matrix = params
    weight_matrix2 = torch.square(weight_matrix)
    iterations = 0
    
    weight_matrix2 = torch.pow(weight_matrix, 2)
    weight_matrix3 = torch.pow(weight_matrix, 3)
    while True:
        mv_prev = torch.clone(mv)
        mh_prev = torch.clone(mh)
        
        dmv = mv - torch.square(mv)
        dmh = mh - torch.square(mh)
        
        field_h = hbias \
            + (mv @ weight_matrix) \
            + (0.5 - mh) * (dmv @ weight_matrix2) \
            + (1/3 - 2 * dmh) * ((dmv * (0.5 - mv)) @ weight_matrix3)
        mh = rho * mh_prev + (1. - rho) * torch.sigmoid(field_h)
        
        dmh = mh - torch.square(mh)
        field_v = vbias \
            + (mh @ weight_matrix.mT) \
            + (0.5 - mv) * (dmh @ weight_matrix2.mT) \
            + (1/3 - 2 * dmv) * ((dmh * (0.5 - mh)) @ weight_matrix3.mT)
        mv = rho * mv_prev + (1. - rho) * torch.sigmoid(field_v)
        eps1 = torch.abs(mv - mv_prev).max()
        eps2 = torch.abs(mh - mh_prev).max()
        if (eps1 < alpha) and (eps2 < alpha):
            break
        iterations += 1
        if iterations >= max_iter:
            break
    return (mv, mh)

def iterate_mean_field(X : Tuple[Tensor, Tensor], params : Tuple[Tensor, Tensor, Tensor], order : int=2,
                       batch_size : int=128, alpha : float=1e-6, verbose : bool=True, 
                       rho : float=0., max_iter : int=2000, device : torch.device=torch.device("cpu")) -> Tuple[Tensor, Tensor]:
    """Iterates the mean field self-consistency equations at the specified order, starting from the visible units X, until convergence.
    Args:
        X (Tuple[Tensor, Tensor]): Initial conditions (visible and hidden magnetizations).
        params (Tuple[Tensor, Tensor, Tensor]): Parameters of the model (vbias, hbias, weight_matrix).
        order (int, optional): Order of the expansion (1, 2, 3). Defaults to 2.
        batch_size (int, optional): Number of samples in each batch. To set based on the memory availability. Defaults to 100.
        alpha (float, optional): Convergence threshold. Defaults to 1e-6.
        verbose (bool, optional): Whether to print the progress bar or not. Defaults to True.
        rho (float, optional): Dumping parameter. Defaults to 0..
        max_iter (int, optional): Maximum number of iterations. Defaults to 2000.
        device (torch.device, optional): Device. Defaults to cpu.
    Raises:
        NotImplementedError: If the specifiend order of expansion has not been implemented.
    Returns:
        Tuple[Tensor, Tensor]: Fixed points of (visible magnetizations, hidden magnetizations)
    """
    if order not in [1, 2, 3]:
        raise NotImplementedError('Possible choices for the order parameter: (1, 2, 3)')
    if order == 1:
        sampling_function = iterate_mf1
    elif order == 2:
        sampling_function = iterate_mf2
    elif order == 3:
        sampling_function = iterate_mf3
    
    n_data = X[0].shape[0]
    mv = torch.tensor([], device=device)
    mh = torch.tensor([], device=device)
    num_batches = n_data // batch_size
    num_batches_tail = num_batches
    if n_data % batch_size != 0:
        num_batches_tail += 1
        if verbose:
            pbar = tqdm(total=num_batches + 1, colour='red', ascii="-#")
            pbar.set_description('Iterating Mean Field')
    else:
        if verbose:
            pbar = tqdm(total=num_batches, colour='red', ascii="-#")
            pbar.set_description('Iterating Mean Field')
    for m in range(num_batches):
        X_batch = []
        for mag in X:
            X_batch.append(mag[m * batch_size : (m + 1) * batch_size, :])

        mv_batch, mh_batch = sampling_function(X_batch, params, alpha=alpha, rho=rho, max_iter=max_iter)
        mv = torch.cat([mv, mv_batch], 0)
        mh = torch.cat([mh, mh_batch], 0)
        
        if verbose:
            pbar.update(1)
    # handle the remaining data
    if n_data % batch_size != 0:
        X_batch = []
        for mag in X:
            X_batch.append(mag[num_batches * batch_size:, :])
            
        mv_batch, mh_batch = sampling_function(X_batch, params, alpha=alpha, rho=rho, max_iter=max_iter)
        mv = torch.cat([mv, mv_batch], 0)
        mh = torch.cat([mh, mh_batch], 0)
        
        if verbose:
            pbar.update(1)
            pbar.close()
    return (mv, mh)