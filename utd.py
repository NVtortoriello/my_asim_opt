import numpy as np

import torch
import torch.nn as nn
from scipy.special import fresnel

def generic_3d_rotation(v, e, theta):
    # Normalize the axis
    e = e / np.linalg.norm(e)
    
    # Compute components
    v_cos_theta = v * np.cos(theta)
    cross = np.cross(e, v) * np.sin(theta)
    dot = np.dot(e, v) * (1 - np.cos(theta)) * e
    
    # Apply Rodrigues' rotation formula
    v_rotated = v_cos_theta + cross + dot
    return v_rotated

def W_rotation_matrix(a , b, q , r):
    W = np.zeros((2,2))
    W[0,0] = np.dot(a , q)
    W[0,1] = np.dot(a , r)
    W[1,0] = np.dot(b , q)
    W[1,1] = np.dot(b , r)
    return W

def a(beta , nu , plus):
    if plus == 1:
        N = np.round((beta + np.pi)/(2*nu * np.pi))
    else:
        N = np.round((beta - np.pi)/(2*nu * np.pi))
        
    return (2 * np.cos((2 * nu* np.pi * N - beta)/2)**2)

def diff_coeff(x):
    S, C = fresnel(np.sqrt(2*x/np.pi))
    F = np.sqrt(np.pi * x/2 ) * np.exp(1j * x) * (1 + 1j - 2*(S + 1j*C))

    return F


#Defined for nu-face
def gtd_diffraction(inc, dif, edg ,nor_0, n, er):
    """ 
    inc: Inc ray direction vector.
    dif: Dif  ray direction vector.
    normal_face : ndarray
        Normal vector to the face of interest.
    edg: Direction vector of the edge.
    nu: Index or identifier for the face of interest. 
    """
    s_o=1       #Distance origin--->Qd
    s_p=1       #Distance Qd ----> obs point
    
    #proj ray_inc onto the plane orthogonal to inc and edge
    inc_t = inc - (np.dot(inc, edg)) * edg
    inc_t /= np.linalg.norm(inc_t)

    #proj ray_dif onto the plane orthogonal to inc and edge
    dif_t = dif - (np.dot(dif, edg)) * edg
    dif_t /= np.linalg.norm(dif_t)

    t0 = np.cross(nor_0, edg)                  #tangent to the current face
    #azimuth computed from the face 0
    phi_i_t0 = np.pi - (np.pi - np.arccos(- np.dot(inc_t, t0)) ) * np.sign(- np.dot(inc_t, nor_0))
    phi_d_t0 = np.pi - (np.pi - np.arccos(  np.dot(dif_t, t0)) ) * np.sign(+ np.dot(dif_t, nor_0))
    
    #angle with normal
    theta_r0 = np.arccos(np.abs(np.sin(phi_i_t0)))
    theta_rn = np.arccos(np.abs(np.sin(n*np.pi - phi_d_t0)))

    #Beta0 angle between dif ray and edge
    beta0 = np.arccos(np.abs(np.dot(dif, edg)))

    #S IS EXPRESSED AS NUMBER OF LAMBDAS
    k= 2*np.pi # /lbda      
    L  = k * (s_o * s_p)/(s_o + s_p) * np.sin(beta0)**2

    #D1 coefficient
    w = -np.exp(-1j * np.pi/4) / (2*n *np.sqrt(2*np.pi * k) * np.sin(beta0))
    x1 =  L * a((phi_d_t0 - phi_i_t0) , n , 1)
    D1 =  w * 1/(np.tan((np.pi + ( phi_d_t0 - phi_i_t0)) / 2/n)) * diff_coeff(x1) 
    #D2 coefficient
    x2 =  L * a((phi_d_t0 - phi_i_t0) , n , 0)
    D2 =  w * 1/(np.tan((np.pi + (-phi_d_t0 + phi_i_t0)) / 2/n)) * diff_coeff(x2) 

    #D3 coefficient
    x3 =  L * a((phi_d_t0 + phi_i_t0) , n , 1)
    D3 =  w * 1/(np.tan((np.pi + (+phi_d_t0 + phi_i_t0)) / 2/n) )* diff_coeff(x3) 

    #D4 coefficient
    x4 =  L * a((phi_d_t0 + phi_i_t0) , n , 0)
    #da capire perche va a0
    D4 =  w * 1/(np.tan((np.pi - (phi_d_t0 + phi_i_t0)) / 2/n)) * diff_coeff(x4) 

    # nu dependt = R matrices of coeffs
    R0 = np.zeros((2,2), dtype=complex)
    Rn = np.zeros((2,2), dtype=complex)

    #Inc spherical coordinates
    theta_i = np.arccos(inc[2])
    phi_i = np.arctan2(inc[1], inc[0])
    
    vec_theta_i = np.array([np.cos(theta_i) * np.cos(phi_i), np.cos(theta_i) * np.sin(phi_i), - np.sin(theta_i)])
    vec_phi_i = np.array([-np.sin(phi_i), np.cos(phi_i),0])
    
    #Intermediate orthogonal dyad efore tm te conversion
    phi_prime_hat = np.cross(inc, edg)
    phi_prime_hat /= np.linalg.norm(phi_prime_hat,2)

    b0_hat = np.cross(phi_prime_hat, inc)
    b0_hat /= np.linalg.norm(b0_hat,2)

    phi_hat = - np.cross(dif, edg)
    phi_hat /= np.linalg.norm(phi_hat,2)
    
    #DIFF spherical coordinates
    theta_r = np.arccos(dif[2])
    phi_r = np.arctan2(dif[1], dif[0])
    
    vec_theta_r = np.array([np.cos(theta_r) * np.cos(phi_r), np.cos(theta_r) * np.sin(phi_r), - np.sin(theta_r)])
    vec_phi_r = np.array([-np.sin(phi_r), np.cos(phi_r),0])  

    for nu in [0,n]:
        nor_nu = generic_3d_rotation(nor_0, edg , nu * np.pi)

        e_pei = np.cross(inc, nor_nu)       #E_perpend to the normal and ray plane TE
        e_pei /= np.linalg.norm(e_pei,2)

        e_pai = np.cross(e_pei, inc)        #E_parallel to the normal and ray plane TM
        e_pai /= np.linalg.norm(e_pai,2)

        e_per = e_pei                       #TM
        e_par = np.cross(e_per, inc)        #E_parallel to the normal and ray plane TE
        e_par /= np.linalg.norm(e_par,2)

        m_i1 = W_rotation_matrix(phi_hat,  b0_hat, vec_theta_i, vec_phi_i)
        m_i2 = W_rotation_matrix(e_pai,e_pei, phi_hat, b0_hat)

        m_r1 = W_rotation_matrix(phi_hat, b0_hat, e_par, e_per)
        m_r2 = W_rotation_matrix(vec_theta_r,vec_phi_r, phi_hat, b0_hat)

        r = np.zeros((2,2), dtype=complex)

        if nu == 0:
            r[0,0] = (er * np.cos(theta_r0) - np.sqrt(er - np.sin(theta_r0)**2)) / (er * np.cos(theta_r0) + np.sqrt(er - np.sin(theta_r0)**2))
            r[1,1] = (np.cos(theta_r0) - np.sqrt(er - np.sin(theta_r0)**2)) / (np.cos(theta_r0) + np.sqrt(er - np.sin(theta_r0)**2))
            R0 = m_r2 @ m_r1 @ r @ m_i2 @ m_i1

        else:            
            r[0,0] = (er * np.cos(theta_rn) - np.sqrt(er - np.sin(theta_rn)**2)) / (er * np.cos(theta_rn) + np.sqrt(er - np.sin(theta_rn)**2))
            r[1,1] = (np.cos(theta_rn) - np.sqrt(er - np.sin(theta_rn)**2)) / (np.cos(theta_rn) + np.sqrt(er - np.sin(theta_rn)**2))
            
            Rn = m_r2 @ m_r1 @ r @ m_i2 @ m_i1

    D = np.zeros((2,2))
    D = - ((D1 + D2) * np.eye(2,2) - D3 * Rn - D4 * R0) * np.sqrt(1/(s_o*s_p*(s_p + s_o))) * np.exp(-1j * k *(s_p + s_o))
    return D

def r_dyad(inc, dep, normal, er):
    #INPUT = CARTESIAN global coordinates
    #OUTPUT = conversion matrix   generic incident electric field ---> TM TE of incident ---> TM TE reflected ---> generic reflected electric field
    
    # INCIDENT part
    # TM polarization
    # H_TM = np.cross(normal, inc)
    e_pai = np.cross(inc, np.cross(normal, inc))        #E_parallel to the incidence plane
    e_pai /= np.linalg.norm(e_pai,2)
    # TE polarization
    e_pei = np.cross(normal, e_pai)                     #E_perpendicular
    e_pei /= np.dot(normal, inc)
    
    # REFLECTED part
    # TM
    e_par = np.cross(dep, np.cross(normal, dep))
    e_par /= np.linalg.norm(e_par,2)

    #reflected TE = identical
    e_per = e_pei

    #Inc spherical coordinates
    theta_i = np.arccos(inc[2])
    phi_i = np.arctan2(inc[1], inc[0])

    #Vectorial expression cartesian-->spherical global
    vec_theta_i = np.array([np.cos(theta_i) * np.cos(phi_i), np.cos(theta_i) * np.sin(phi_i), - np.sin(theta_i)])
    vec_phi_i = np.array([-np.sin(phi_i), np.cos(phi_i),0])
    
    #Reflected spherical coordinates
    theta_r = np.arccos(dep[2])
    phi_r = np.arctan2(dep[1], dep[0])
    
    vec_theta_r = np.array([np.cos(theta_r) * np.cos(phi_r), np.cos(theta_r) * np.sin(phi_r), - np.sin(theta_r)])
    vec_phi_r = np.array([-np.sin(phi_r), np.cos(phi_r),0])

    # print(f'e_pai e_pad {np.sign(np.dot(e_pai, e_pad))}')
    #Matrix to convert incident field given by two orthogonal polarizations ALIGNED WITH THETA AND PHI GLOBAL
    #into TM and TE polarizations
    m_i = W_rotation_matrix(e_pai,e_pei, vec_theta_i, vec_phi_i)

    #*m_i trasforma la coppia (theta_hat, phi_hat) inc --->(TE , TM)    
    #Matrix to convert REFLECTED field given by two orthogonal polarizations ALIGNED WITH THETA AND PHI GLOBAL
    #into TM and TE polarizations    
    m_r = W_rotation_matrix(vec_theta_r,vec_phi_r, e_par, e_per)
    #*m_r trasforma la coppia (TE , TM) ref--->(theta_hat, phi_hat) reflecteed
    alpha = np.pi-np.arccos(np.dot(normal, inc))

    r = np.zeros((2,2), dtype=complex)

    #FRESNELL EQUATION WHEN n1 IS VACUUM
    r[0,0] = (er * np.cos(alpha) - np.sqrt(er - np.sin(alpha)**2)) / (er * np.cos(alpha) + np.sqrt(er - np.sin(alpha)**2))
    r[1,1] = (np.cos(alpha) - np.sqrt(er - np.sin(alpha)**2)) / (np.cos(alpha) + np.sqrt(er - np.sin(alpha)**2))

    #transmission
    trs =  (1 / np.sqrt(er)) * inc + (1 / np.sqrt(er) * np.cos(alpha) - np.sqrt(1 - (1 / er) * np.sin(alpha)**2)) * normal
    trs =  np.real(trs)
    trs /= np.linalg.norm(trs) 

    t = np.zeros((2,2), dtype=complex)
    t[1,1] = (2*np.cos(alpha))/(np.cos(alpha) + np.sqrt(er - np.sin(alpha)**2 ))
    t[0,0] = (2*np.sqrt(er)* np.cos(alpha))/(er* np.cos(alpha) + np.sqrt(er - np.sin(alpha)**2 ))


    # TRANSMITTED part-TM
    e_pat = np.cross(trs, np.cross(normal, trs))
    e_pat /= np.linalg.norm(e_pat,2)
    #TE = identical
    e_pet = e_pei
     
    theta_t = np.arccos(trs[2])
    phi_t= np.arctan2(trs[1], trs[0])
    
    vec_theta_t = np.array([np.cos(theta_t) * np.cos(phi_t), np.cos(theta_t) * np.sin(phi_t), - np.sin(theta_t)])
    vec_phi_t = np.array([-np.sin(phi_t), np.cos(phi_t),0])

    m_t = W_rotation_matrix(vec_theta_t,vec_phi_t, e_pat, e_pet)


    result = (m_r @ r @ m_i)  + (m_t @ t @ m_i)
    # result = (m_r @ r @ m_i)   
    # result = (m_t @ t @ m_i)

    return result , trs

class torch_r_dyad(nn.Module):

    def __init__(self):
        super(torch_r_dyad, self).__init__()
        self.eps =  nn.Parameter(torch.ones(1))
        self.conductivity =  nn.Parameter(torch.ones(1))

    def forward(self, x):

        inc = x[0]
        dep = x[1]
        nor = x[2]
        trs = x[3]

        output = torch.zeros((inc.shape[0], 2, 2),dtype=torch.complex64)

        # c = 299792458
        f = 3.6e9
        # wl = c / f
        # wv = 2 * np.pi / wl

        e0 = 8.8541878188e-12

        for idx in range(inc.shape[0]):

            e_pai = torch.cross(inc[idx], torch.cross(nor[idx], inc[idx], dim=-1), dim=-1)
            e_pai /= torch.linalg.norm(e_pai, 2)

            e_pei = torch.cross(nor[idx], e_pai, dim=-1) 
            e_pei /= torch.dot(nor[idx], inc[idx])
            
            e_par = torch.cross(dep[idx], torch.cross(nor[idx], dep[idx],dim=-1),dim=-1)
            e_par /= torch.linalg.norm(e_par,2)

            e_per = e_pei

            theta_i = torch.arccos(inc[idx][2])
            phi_i = torch.arctan2(inc[idx][1], inc[idx][0])
            
            vec_theta_i = torch.tensor([np.cos(theta_i) * np.cos(phi_i), np.cos(theta_i) * np.sin(phi_i), - np.sin(theta_i)])
            vec_phi_i = torch.tensor([-np.sin(phi_i), np.cos(phi_i),0])
            
            theta_r = torch.arccos(dep[idx][2])
            phi_r = torch.arctan2(dep[idx][1], dep[idx][0])
            
            vec_theta_r = torch.tensor([np.cos(theta_r) * np.cos(phi_r), np.cos(theta_r) * np.sin(phi_r), - np.sin(theta_r)])
            vec_phi_r = torch.tensor([-np.sin(phi_r), np.cos(phi_r),0])
            
            m_i = torch.zeros((2,2),dtype=torch.complex64)

            m_i[0,0] = torch.dot(e_pai, vec_theta_i)
            m_i[0,1] = torch.dot(e_pai, vec_phi_i)
            m_i[1,0] = torch.dot(e_pei, vec_theta_i)
            m_i[1,1] = torch.dot(e_pei, vec_phi_i)

            m_r = torch.zeros((2,2),dtype=torch.complex64)
        
            m_r[0,0] = torch.dot(vec_theta_r, e_par)
            m_r[0,1] = torch.dot(vec_theta_r, e_per)
            m_r[1,0] = torch.dot(vec_phi_r, e_par)
            m_r[1,1] = torch.dot(vec_phi_r, e_per)
            
            alpha = np.pi-torch.arccos(torch.dot(nor[idx], inc[idx]))

            r = torch.zeros((2,2),dtype=torch.complex64)

            er = torch.complex(self.eps, - self.conductivity / 2 / torch.pi / f / e0) 

            r[0,0] = (er * torch.cos(alpha) - torch.sqrt(er - torch.float_power(torch.sin(alpha),2))) / (er * torch.cos(alpha) + torch.sqrt(er - torch.float_power(torch.sin(alpha),2)))
            r[1,1] = (torch.cos(alpha) - torch.sqrt(er - torch.float_power(torch.sin(alpha),2))) / (torch.cos(alpha) + torch.sqrt(er - torch.float_power(torch.sin(alpha),2)))


            #transmission
            t = torch.zeros((2,2),dtype=torch.complex64)
            t[1,1] = (2*torch.cos(alpha))/(torch.cos(alpha) + torch.sqrt(er - torch.sin(alpha)**2 ))
            t[0,0] = (2*torch.sqrt(er)* torch.cos(alpha))/(er* torch.cos(alpha) + torch.sqrt(er - torch.sin(alpha)**2 ))


            # TRANSMITTED part-TM
            e_pat = torch.cross(trs[idx], torch.cross(nor[idx], trs[idx] ,dim=-1),dim=-1)
            e_pat /= torch.linalg.norm(e_pat,2)
            #TE = identical
            e_pet = e_pei
            
            theta_t = torch.arccos(trs[idx][2])
            phi_t= torch.arctan2(trs[idx][1], trs[idx][0])
            
            vec_theta_t = torch.tensor([np.cos(theta_t) * np.cos(phi_t), np.cos(theta_t) * np.sin(phi_t), - np.sin(theta_t)])
            vec_phi_t = torch.tensor([-np.sin(phi_t), np.cos(phi_t),0])
            
            m_t = torch.zeros((2,2),dtype=torch.complex64)
            m_t[0,0] = torch.dot(vec_theta_t, e_pat)
            m_t[0,1] = torch.dot(vec_theta_t, e_pet)
            m_t[1,0] = torch.dot(vec_phi_t, e_pat)
            m_t[1,1] = torch.dot(vec_phi_t, e_pet)

            result = torch.matmul(m_r, torch.matmul(r, m_i)) 
            result = result + torch.matmul(m_t, torch.matmul(t, m_i))

            output[idx,:,:] = result

        return output
