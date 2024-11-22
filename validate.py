import numpy as np

from utils import unit_vector, rect_normal

def val_r(tx, rxs, rfs, normal):

    for idx, rx in enumerate(rxs):

        imp = unit_vector(tx.point, rfs[idx])
        dep = unit_vector(rfs[idx], rx.point)

        cos_theta_i = np.dot(-imp, normal)
        cos_theta_r = np.dot(normal, dep)

        if not np.isclose(np.abs(cos_theta_i - cos_theta_r), 0):
            print(f'{cos_theta_i} {cos_theta_r}')
            raise ValueError
