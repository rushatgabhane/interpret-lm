import numpy as np


def dot_product(s, e):
    """L1-normalise saliency then dot with binary gold mask."""
    if s.sum() == 0:
        return 0.0
    s_norm = s / s.sum()
    return float(np.dot(s_norm, e))


def probes_needed(s, e):
    return int(np.argsort(s * -1)[np.where(e == 1)[0][0]]) + 1


def reciprocal_rank(s, e):
    return 1 / probes_needed(s, e)
