import numpy as np

def l2_dist(p, c):
    return np.square(p - c).mean()