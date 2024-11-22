import numpy as np

import torch
import torch.nn as nn


def r_dyad(inc, dep, normal, er):
    
    e_pai = np.cross(inc, np.cross(normal, inc))
    e_pai /= np.linalg.norm(e_pai,2)
    
    e_pei = np.cross(normal, e_pai) 
    e_pei /= np.dot(normal, inc)
    
    e_par = np.cross(dep, np.cross(normal, dep))
    e_par /= np.linalg.norm(e_par,2)

    e_per = e_pei

    theta_i = np.arccos(inc[2])
    phi_i = np.arctan2(inc[1], inc[0])
    
    vec_theta_i = np.array([np.cos(theta_i) * np.cos(phi_i), np.cos(theta_i) * np.sin(phi_i), - np.sin(theta_i)])
    vec_phi_i = np.array([-np.sin(phi_i), np.cos(phi_i),0])
    
    theta_r = np.arccos(dep[2])
    phi_r = np.arctan2(dep[1], dep[0])
    
    vec_theta_r = np.array([np.cos(theta_r) * np.cos(phi_r), np.cos(theta_r) * np.sin(phi_r), - np.sin(theta_r)])
    vec_phi_r = np.array([-np.sin(phi_r), np.cos(phi_r),0])
    
    m_i = np.zeros((2,2))

    # print(f'e_pai e_pad {np.sign(np.dot(e_pai, e_pad))}')
    
    m_i[0,0] = np.dot(e_pai, vec_theta_i)
    m_i[0,1] = np.dot(e_pai, vec_phi_i)
    m_i[1,0] = np.dot(e_pei, vec_theta_i)
    m_i[1,1] = np.dot(e_pei, vec_phi_i)

    m_r = np.zeros((2,2))
   
    m_r[0,0] = np.dot(vec_theta_r, e_par)
    m_r[0,1] = np.dot(vec_theta_r, e_per)
    m_r[1,0] = np.dot(vec_phi_r, e_par)
    m_r[1,1] = np.dot(vec_phi_r, e_per)
    
    alpha = -np.arccos(np.dot(normal, inc))

    r = np.zeros((2,2), dtype=complex)
    r[0,0] = (er * np.cos(alpha) - np.sqrt(er - np.sin(alpha)**2)) / (er * np.cos(alpha) + np.sqrt(er - np.sin(alpha)**2))
    r[1,1] = (np.cos(alpha) - np.sqrt(er - np.sin(alpha)**2)) / (np.cos(alpha) + np.sqrt(er - np.sin(alpha)**2))

    # print(r)

    result = m_r @ r @ m_i

    return result

class torch_r_dyad(nn.Module):

    def __init__(self, normal):
        super(torch_r_dyad, self).__init__()

        self.normal = normal