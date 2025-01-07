import numpy as np
from copy import deepcopy

class sim_wall:

    def __init__(self, x1, x3, eps, sigma):

        self.x1 = deepcopy(x1)  # bottom left
        self.x2 = deepcopy(x3)
        self.x3 = deepcopy(x3)
        self.x4 = deepcopy(x1)
        
        self.x2[2] = self.x1[2]
        self.x4[2] = self.x3[2]

        self.eps = eps
        self.sigma = sigma

    def __str__(self):
        buffer = f'x1: {self.x1}, x2: {self.x2}, x3: {self.x3}, x4: {self.x4}, er: {self.eps}, sigma: {self.sigma}'
        return buffer


class sim_em_ap:

    def __init__(self, x, az, zt, power):

        self.point = x
        self.b_az = az
        self.b_zt = zt
        self.power = power

    def __str__(self):
        buffer = f'point: {self.point}, az: {self.b_az}, zt: {self.b_zt}'
        return buffer
    

def mirrored_point(ap, rect):
    
    edge1 = rect.x2 - rect.x1
    edge2 = rect.x3 - rect.x1
    
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    
    rect_point = rect.x1
    to_point = ap.point - rect_point
    distance_to_plane = np.dot(to_point, normal)
    projection = ap.point - distance_to_plane * normal
    
    mirrored_point = projection - (ap.point - projection)
    
    return mirrored_point


def ray_intersection(ray_origin, ray_direction, rect):
    
    edge1 = rect.x2 - rect.x1
    edge2 = rect.x3 - rect.x1
    
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    
    denom = np.dot(normal, ray_direction)
    if np.isclose(denom, 0):
        return None
    
    d = np.dot(normal, rect.x1)
    t = (d - np.dot(normal, ray_origin.point)) / denom
    
    if t < 0:
        return None
    
    intersection = ray_origin.point + t * ray_direction
    
    p = intersection - rect.x1
    u = np.dot(p, edge1) / np.dot(edge1, edge1)
    v = np.dot(p, edge2) / np.dot(edge2, edge2)
    
    if 0 <= u <= 1 and 0 <= v <= 1:
        return intersection
    
    return None
