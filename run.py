import numpy as np
import matplotlib.pyplot as plt

from transforms import transform_g_to_l, transform_l_to_g

from utils import unit_vector, rect_normal, point_distance, poly_length
from sim import sim_em_ap, sim_wall, ray_intersection, mirrored_point
from validate import val_r

from batch import train

if __name__ == "__main__":

    wall = sim_wall(np.array([0,-5,0]), np.array([0,5,10]), eps=3, sigma=0.01)
    tx = sim_em_ap(np.array([2, -7, 1.5]), 0, 0, 1)

    rxs = []

    for dy in np.arange(0,1,0.1):
        rxs.append(sim_em_ap(np.array([2.5, 4 +dy, 1.5]), 0, 0, 0))

    rfs = []

    for rx in rxs:
        mr_point = mirrored_point(rx, wall)
        ray_direction = unit_vector(tx.point, mr_point)
        hit_point = ray_intersection(tx, ray_direction, wall)

        rfs.append(hit_point)

    normal = rect_normal(wall)

    try:
        val_r(tx, rxs, rfs, normal)
    except ValueError:
        print("Something went wrong")
    else:
        print("Validation completed successfully")

    train(tx, rfs, rxs, normal, wall.eps, wall.sigma)
