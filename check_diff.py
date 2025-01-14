import numpy as np

from utils import unit_vector, rect_normal
from sim import sim_em_ap, sim_wall, ray_intersection, mirrored_point
from transforms import transform_l_to_g 
from utils import unit_vector, point_distance
from utd import r_dyad , gtd_diffraction
# from validate import val_r
 

if __name__ == "__main__":  

    SIGMA = 1e5
    EPS = 3

    #EDGE
    e = np.array([1 , 0 , 0])

    tx_pos=[4.5, 7, 1.5]
    tx = sim_em_ap(np.array(tx_pos), 0, 0, 1)

    field = np.sqrt(tx.power) * np.array([complex(0,1), complex(0,0)])

    #Find Qd
    Qd = np.array([0 , 0 , 0])
    normal_fac0 = np.array([0 , 0 , 1])
    n=1.7

    ray_direction = unit_vector(tx.point, Qd) 
    b0 = np.arccos(np.abs( np.dot(e , ray_direction) ) )
 
    rx_pos = np.array([4.5 ,-7 , 1.5 ])
    rx = sim_em_ap(rx_pos, 0, 0, 0)
    # ray_direction = unit_vector(rx.point, Qd) 
    # b0s = np.arccos(np.abs( np.dot(e , ray_direction) ) )
  
    c = 299792458
    f = 3.6e9
    wl = c / f
    wv = 2 * np.pi / wl

    e0 = 8.8541878188e-12
    mu0 = 4*np.pi*1e-7

    conductivity = SIGMA
    er = complex(EPS, - conductivity / 2 / np.pi / f / e0)  

    ifield = transform_l_to_g(field, tx.b_az, tx.b_zt, 0, unit_vector(tx.point, Qd))

    pre_radius = point_distance(tx.point, Qd)
    delta = point_distance(Qd, rx.point)
    pst_radius = pre_radius + delta


    inc = unit_vector(tx.point, Qd)
    dep = unit_vector(Qd, rx.point)

    # m_r, trs = r_dyad(inc, dep, normal, er)

    gtd_diffraction(inc , dep, e, normal_fac0 , n , EPS)
 