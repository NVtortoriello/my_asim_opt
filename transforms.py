import numpy as np


def rot(alpha, beta, gamma):

    r_alpha = np.zeros((3,3))

    r_alpha[0,0] = np.cos(alpha)
    r_alpha[0,1] = -np.sin(alpha)
    r_alpha[0,2] = 0
    r_alpha[1,0] = np.sin(alpha)
    r_alpha[1,1] = np.cos(alpha)
    r_alpha[1,2] = 0
    r_alpha[2,0] = 0
    r_alpha[2,1] = 0
    r_alpha[2,2] = 1

    r_beta = np.zeros((3,3))

    r_beta[0,0] = np.cos(beta)
    r_beta[0,1] = 0
    r_beta[0,2] = np.sin(beta)
    r_beta[1,0] = 0
    r_beta[1,1] = 1
    r_beta[1,2] = 0
    r_beta[2,0] = -np.sin(beta)
    r_beta[2,1] = 0
    r_beta[2,2] = np.cos(beta)
    
    r_gamma = np.zeros((3,3))

    r_gamma[0,0] = 1
    r_gamma[0,1] = 0
    r_gamma[0,2] = 0
    r_gamma[1,0] = 0
    r_gamma[1,1] = np.cos(gamma)
    r_gamma[1,2] = -np.sin(gamma)
    r_gamma[2,0] = 0
    r_gamma[2,1] = np.sin(gamma)
    r_gamma[2,2] = np.cos(gamma)

    m_r = r_alpha @ r_beta @ r_gamma

    return m_r


def transform_l_to_g(field, az, zt, rt, vec):
    #vec is the LOCATION where we evaluate the field
    #field is the 2 components field

    #az: alpha, rotation around z
    #zt rotation around y
    #rt rotation round x
    
    r = rot(az, zt, rt).T # global to local rotation matrix(transpose is local --> global)

    #LOCAL angles
    theta = np.arccos(vec[2])
    phi = np.arctan2(vec[1], vec[0])
    
    #buffer : turn [0 0 1] into a row EXPLICITLY
    #buffer : turn [0 0 1] into a col EXPLICITLY
    #buffer creates a SCALAR argument to retrieve theta GLOBAL
    buffer = (np.expand_dims(np.array([0,0,1]),0) @ r @ np.expand_dims(vec,-1))[0,0]
    theta_p = np.arccos(buffer)

    #buffer creates a SCALAR argument to retrieve phi GLOBAL
    buffer = (np.expand_dims(np.array([complex(1,0),complex(0,1),complex(0,0)]),0) @ r @ np.expand_dims(vec,-1))[0,0]
    phi_p = np.arctan2(buffer.imag, buffer.real)

    #once you have both angles for both systems
    #theta hat as vector function of x y z
    #phi hat as vector function of x y z
    vec_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), - np.sin(theta)])
    vec_phi = np.array([-np.sin(phi), np.cos(phi),0])

    vec_theta_p = np.array([np.cos(theta_p) * np.cos(phi_p), np.cos(theta_p) * np.sin(phi_p), - np.sin(theta_p)])
    vec_phi_p = np.array([-np.sin(phi_p), np.cos(phi_p),0])

    #it s a 2x2 because it s a two components field to be rotated
    t = np.zeros((2,2))

    t[0,0] = (np.expand_dims(vec_theta,0) @ r.T @ vec_theta_p)[0]
    t[0,1] = (np.expand_dims(vec_theta,0) @ r.T @ vec_phi_p)[0]
    t[1,0] = (np.expand_dims(vec_phi,0) @ r.T @ vec_theta_p)[0]
    t[1,1] = (np.expand_dims(vec_phi,0) @ r.T @ vec_phi_p)[0]

    return t @ field


def transform_g_to_l(field, az, zt, rt, vec):

    r = rot(az, zt, rt).T # global to local

    theta = np.arccos(vec[2])
    phi = np.arctan2(vec[1], vec[0])
    
    buffer = (np.expand_dims(np.array([0,0,1]),0) @ r @ np.expand_dims(vec,-1))[0,0]
    theta_p = np.arccos(buffer)
    buffer = (np.expand_dims(np.array([1,complex(0,1),0]),0) @ r @ np.expand_dims(vec,-1))[0,0]
    phi_p = np.arctan2(buffer.imag, buffer.real)

    vec_theta = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), - np.sin(theta)])
    vec_phi = np.array([-np.sin(phi), np.cos(phi),0])

    vec_theta_p = np.array([np.cos(theta_p) * np.cos(phi_p), np.cos(theta_p) * np.sin(phi_p), - np.sin(theta_p)])
    vec_phi_p = np.array([-np.sin(phi_p), np.cos(phi_p),0])

    t = np.zeros((2,2))

    t[0,0] = (np.expand_dims(vec_theta_p,0) @ r @ np.expand_dims(vec_theta,-1))[0,0]
    t[0,1] = (np.expand_dims(vec_theta_p,0) @ r @ np.expand_dims(vec_phi,-1))[0,0]
    t[1,0] = (np.expand_dims(vec_phi_p,0) @ r @ np.expand_dims(vec_theta,-1))[0,0]
    t[1,1] = (np.expand_dims(vec_phi_p,0) @ r @ np.expand_dims(vec_phi,-1))[0,0]

    return t @ field
