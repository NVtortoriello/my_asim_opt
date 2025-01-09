import numpy as np

from utils import unit_vector, rect_normal
from sim import sim_em_ap, sim_wall, ray_intersection, mirrored_point
from batch import generate_batch
# from validate import val_r

from batch import train

if __name__ == "__main__":

    wall = sim_wall(np.array([0,-5,0]), np.array([0,5,10]), eps=0.8, sigma=0.01)
    tx = sim_em_ap(np.array([2, -7, 1.5]), 0, 0, 1)
    
    #Number of received rays
    rxs = []

    for dy in np.arange(0,1,0.1):
        rxs.append(sim_em_ap(np.array([2.5, 4 +dy, 1.5]), 0, 0, 0))
   
    #point of the wall hit by rays
    rfs = []

    for rx in rxs:
        #1) mirror the receiving point
        mr_point = mirrored_point(rx, wall)
        #2) find exact ray from transmitted
        ray_direction = unit_vector(tx.point, mr_point)
        #3) find point on the wall(notice the transmitter)
        hit_point = ray_intersection(tx, ray_direction, wall)

        rfs.append(hit_point)

    normal = rect_normal(wall)

    # try:
    #     val_r(tx, rxs, rfs, normal)
    # except ValueError:
    #     print("Something went wrong")
    # else:
    #     print("Validation completed successfully")

    batch = generate_batch(tx, rfs, rxs, normal, wall.eps, wall.sigma)
