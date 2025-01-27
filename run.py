import numpy as np

from utils import unit_vector, rect_normal
from sim import sim_em_ap, sim_wall, ray_intersection, mirrored_point
# from validate import val_r

from batch import train

if __name__ == "__main__":
    Default = 1

    if Default:
        x_bot_left =[0,-5,0]
        x_top_right=[0,5,10]
        tx_pos=[2, -7, 1.5]
        rx_pos=np.array([2.5, 4 , 1.5])
    else: 
        x_bot_left =[0,-5,-5]
        x_top_right=[0,5,5]
        tx_pos=[4.5, -7, 0]
        rx_pos=np.array([2, 4 , 0])

    wall = sim_wall(np.array(x_bot_left), np.array(x_top_right), eps=2, sigma=0.01)
    tx = sim_em_ap(np.array(tx_pos), 0, 0, 1)
    
    normal = rect_normal(wall)

    #Number of received rays
    rxs = []

    # for dy in np.arange(0,1,0.1):
    #     rxs.append(sim_em_ap(rx_pos+np.array([0, dy, 0]), 0, 0, 0))
   
    #point of the wall hit by rays
    rfs = []


    #Could it be an option to have rx set inside a unique for so that the non intersected points are not considered?
    # for rx in rxs:
    
    for dy in np.arange(0,1,0.1):
        rx = sim_em_ap(rx_pos+np.array([0, dy, 0]), 0, 0, 0)

        #1) mirror the receiving point: 
        # trova il punto che il raggio colpirebbe se non ci fosse il muro
        mr_point = mirrored_point(rx, wall) 
        #2) find exact ray from transmitter
        ray_direction = unit_vector(tx.point, mr_point)
        #3) find point on the wall
        hit_point = ray_intersection(tx, ray_direction, wall)

        #4)upload the rxs rfs only if hit point exists
        if hit_point is not None:
            rfs.append(hit_point)
            rxs.append(rx)

    # try:
    #     val_r(tx, rxs, rfs, normal)
    # except ValueError:
    #     print("Something went wrong")
    # else:
    #     print("Validation completed successfully")

    train(tx, rfs, rxs, normal, wall.eps, wall.sigma)
