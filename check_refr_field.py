import numpy as np

from utils import unit_vector, rect_normal
from sim import sim_em_ap, sim_wall, ray_intersection, mirrored_point
from transforms import transform_l_to_g 
from utils import unit_vector, point_distance
from utd import r_dyad 

if __name__ == "__main__":  

    x_bot_left =[0,-5,-5]
    x_top_right=[0,5,5]
    tx_pos=[4.5, -7, 0]
    rx_pos=np.array([2, 4 , 0])

    EPSILON = 2
    SIGMA= 0.01

    wall = sim_wall(np.array(x_bot_left), np.array(x_top_right), eps=EPSILON, sigma=SIGMA)
    tx = sim_em_ap(np.array(tx_pos), 0, 0, 1)
    
    normal = rect_normal(wall) 
    rxs = [] 
    rfs = [] 
    for dy in np.arange(0,1,0.1):
        rx = sim_em_ap(rx_pos+np.array([0, dy, 0]), 0, 0, 0)
 
        mr_point = mirrored_point(rx, wall)  
        ray_direction = unit_vector(tx.point, mr_point) 
        hit_point = ray_intersection(tx, ray_direction, wall)

        #4)update the rxs rfs only if hit point exists
        if hit_point is not None:
            rfs.append(hit_point)
            rxs.append(rx)
 
    field = np.sqrt(tx.power) * np.array([complex(0,1), complex(0,0)])

    for idx, rx in enumerate(rxs):

        rf = rfs[idx]

        c = 299792458
        f = 3.6e9
        wl = c / f
        wv = 2 * np.pi / wl

        e0 = 8.8541878188e-12
        mu0 = 4*np.pi*1e-7

        conductivity = wall.sigma
        er = complex(wall.eps, - conductivity / 2 / np.pi / f / e0)  

        ifield = transform_l_to_g(field, tx.b_az, tx.b_zt, 0, unit_vector(tx.point, rf))

        pre_radius = point_distance(tx.point, rf)
        delta = point_distance(rf, rx.point)
        pst_radius = pre_radius + delta


        inc = unit_vector(tx.point, rf)
        dep = unit_vector(rf, rx.point)

        m_r, trs = r_dyad(inc, dep, normal, er)

        #ifield is not used at the moment
        #modulo cambia per effetto di epsilon dielettrico
        Z0 = np.sqrt(1./wall.eps)
        ref_field = m_r @ ifield #* np.exp(complex(0, -wv*pst_radius)) #/ pst_radius /4 /np.pi


        print(f'Rx:{rx.point}   Incident Field : {np.linalg.norm(ifield)} , Refracted field : {np.linalg.norm(ref_field)}')
 
 