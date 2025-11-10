import numpy as np
from .utils import resolve_data_file
from scipy.io import loadmat

def dynamic_change(rng: np.random.Generator, u: np.ndarray, change_type: int, u_min: float, u_max: float, u_severity: float, change_count: int) -> np.ndarray:
    """Apply dynamic change operator replicating MATLAB logic.
    Parameters mirror MATLAB dynamic_change.m. rng is a numpy Generator.
    """
    p = 12
    noisy_severity = 0.8
    A = 3.67
    alpha = 0.04
    alpha_max = 0.1
    u_range = u_max - u_min
    u = u.copy()
    if change_type == 1:  # small step
        r = rng.uniform(-1, 1, size=u.shape[0])
        u = np.clip(u + alpha * u_range * r * u_severity, u_min, u_max)
    elif change_type == 2:  # large step
        r = rng.uniform(-1, 1, size=u.shape[0])
        u = np.clip(u + u_range * (alpha * np.sign(r) + (alpha_max - alpha) * r) * u_severity, u_min, u_max)
    elif change_type == 3:  # random
        u = np.clip(u + rng.standard_normal(size=u.shape[0]) * u_severity, u_min, u_max)
    elif change_type == 4:  # chaotic
        u = (u - u_min) / u_range
        u = A * u * (1 - u)
        u = np.clip(u_min + u * u_range, u_min, u_max)
    elif change_type in (5, 6):  # recurrent (+ noisy)
        # phi.mat contains variable 'phi'
        phi_mat = loadmat(resolve_data_file('phi.mat'))
        phi = phi_mat.get('phi')
        if phi is None:
            raise KeyError('phi variable not found in phi.mat')
        phi = np.ravel(phi)
        phi = phi[:u.shape[0]]
        base = u_min + u_range * (np.sin(2 * np.pi * change_count / p + phi) + 1) / 2
        if change_type == 5:
            u = base
        else:
            u = np.clip(base + rng.standard_normal(size=u.shape[0]) * noisy_severity, u_min, u_max)
    else:
        raise ValueError('change_type has the wrong number.')
    return u
