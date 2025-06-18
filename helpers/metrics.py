import numpy as np


def dot_product(s, e):
    return float((s * e).sum())


def probes_needed(s, e):
    return int(np.argsort(s * -1)[np.where(e == 1)[0][0]]) + 1


def reciprocal_rank(s, e):
    return 1 / probes_needed(s, e)
