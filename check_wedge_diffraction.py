import numpy as np

from utils import unit_vector, rect_normal
from sim import sim_em_ap, sim_wall, sim_wedge_2D, ray_intersection, mirrored_point
from transforms import transform_l_to_g 
from utils import unit_vector, point_distance
from utd import r_dyad , gtd_diffraction
# from validate import val_r
 

if __name__ == "__main__":  

    SIGMA = 0.01 * 1e0
    EPS = 3

    #transmitter
    tx_pos=[-4,4,4]
    tx_pos=[0,4,4]
    tx = sim_em_ap(np.array(tx_pos), 0, 0, 1)
    field = np.sqrt(tx.power) * np.array([complex(0,1), complex(0,0)])
   
    n=1.5
    alpha = (2-n)* np.pi

    #Diffraction point
    edg = np.array([+1.0, 0 , 0])       #height
    edg/= np.linalg.norm(edg) 
    
    n0 = np.array([0 , 0.0 , 1])       #height
    n0/= np.linalg.norm(n0) 
    
    t0 = np.cross(n0,edg)

    Qd = np.array([0 ,0 , 0])
    s_tx = np.linalg.norm(tx.point - Qd)
    s_rx=s_tx
    s_direction = unit_vector(tx.point, Qd)

    b0 = np.arccos(np.abs( np.dot(s_direction  , edg) ) )

    rx_pos = Qd - s_tx * np.cos(b0) * edg + s_tx * np.sin(b0) * n0
    rx_dir =  unit_vector(Qd,rx_pos) 
 
    c = 299792458
    f = 3.6e9
    wl = c / f
    wv = 2 * np.pi / wl

    e0 = 8.8541878188e-12
    mu0 = 4*np.pi*1e-7

    conductivity = SIGMA
    er = complex(EPS, - conductivity / 2 / np.pi / f / e0)  

    ifield = transform_l_to_g(field, tx.b_az, tx.b_zt, 0, unit_vector(tx.point, Qd))

    #Path length  
    pst_radius = s_tx + s_rx 

    inc = unit_vector(tx.point, Qd)
    difr = unit_vector(Qd, rx_pos)  

    if np.arccos(np.abs( np.dot(difr , edg) ) ) == b0:
        print(f'Correct Keller cone check ; b={b0*180/np.pi}')

    D = gtd_diffraction(inc , s_tx/wl, difr, s_tx/wl , edg, n0 , n , er)

    dif_field = D @ ifield

    print(f'Rx:{rx_pos}   Incident Field : {np.linalg.norm(ifield)} , Diffracted Field : {np.linalg.norm(dif_field)}')
 
 