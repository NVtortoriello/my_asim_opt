import numpy as np

from utils import unit_vector, rect_normal

def val_r(tx, rxs, rfs, normal):
    #tx fixed position
    #rfs is reflecting surface position

    for idx, rx in enumerate(rxs):
        #rx is each point receiving vector rxs 

        imp = unit_vector(tx.point, rfs[idx])
        dep = unit_vector(rfs[idx], rx.point)

        cos_theta_i = np.dot(-imp, normal)              #incident ray
        cos_theta_r = np.dot(normal, dep)               #reflected ray

        if not np.isclose(np.abs(cos_theta_i - cos_theta_r), 0):
            print(f'{cos_theta_i} {cos_theta_r}')
            raise ValueError

def val_t(tx, rxs, rfs, normal, er):
    #tx fixed position
    #rfs is reflecting surface position

    for idx, rx in enumerate(rxs):
        #rx is each point receiving vector rxs 

        inc = unit_vector(tx.point, rfs[idx])
        dep = unit_vector(rfs[idx], rx.point)
        alpha = -np.arccos(np.dot(normal, inc))

        trs = (1 / np.sqrt(np.real(er))) * inc + (1 / np.sqrt(np.real(er)) * np.cos(alpha) - np.sqrt(1 - (1 / np.real(er)) * np.sin(alpha)**2)) * normal
   
        cos_theta_i = np.dot(-inc, normal)              #incident ray
        cos_theta_r = np.dot(normal, dep)               #reflected ray

        if not np.isclose(np.abs(cos_theta_i - cos_theta_r), 0):
            print(f'{cos_theta_i} {cos_theta_r}')
            raise ValueError