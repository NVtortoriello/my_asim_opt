import numpy as np
from copy import deepcopy

_
    
    

class sim_wedge_2D:
    #create a rect wall given 2 points

    def __init__(self, x1, edge, n , er, sigma_in):

        self.orig  = deepcopy(x1)  # bottom left
        self.edge  = deepcopy(edge)
        self.edge  /= np.linalg.norm(self.edge)

        self.n = n
        self.eps = er
        self.sigma = sigma_in

    def __str__(self):
        buffer = f'Origin: {self.orig}, edge: {self.edge}, er: {self.eps}, sigma: {self.sigma}, n: {self.n}'
        return buffer
    
class sim_em_ap:

    def __init__(self, x, az, zt, power):

        self.point = x          #location
        self.b_az = az          #tilt?
        self.b_zt = zt          #some angle?
        self.power = power      #power

    def __str__(self):
        buffer = f'point: {self.point}, az: {self.b_az}, zt: {self.b_zt}'
        return buffer
    

def mirrored_point(ap, rect):       #reflection
    
    edge1 = rect.x2 - rect.x1       #width
    # edge2 = rect.x3 - rect.x1       #height
    edge2 = rect.x4 - rect.x1       #height
    
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)  #normal versor

    
    # bottom left is reference
    rect_point = rect.x1
    # vector connecting ray to bottom left
    to_point = ap.point - rect_point
    distance_to_plane = np.dot(to_point, normal)
    projection = ap.point - distance_to_plane * normal

    #(ap.point - projection)= proj--> ap (it s parallel to n )
    mirrored_point = projection - (ap.point - projection)

    return mirrored_point


def ray_intersection(ray_origin, ray_direction, rect):
    #define edges
    edge1 = rect.x2 - rect.x1
    # edge2 = rect.x3 - rect.x1
    edge2 = rect.x4 - rect.x1      #non cambia molto

    #define normal
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)
    #projection of the ray vector on the normal
    denom = np.dot(normal, ray_direction)

    if np.isclose(denom, 0):
        return None
    #projection of bottom left on the normal
    d = np.dot(normal, rect.x1)         
    
    #projection of the ray origin position vector on the normal
    #se perpendicolare avrei t = numeratore perche denom = 1(distanza non influenzata da orientation)
    t = (d - np.dot(normal, ray_origin.point)) / denom
    
    if t < 0:
        return None
    
    #intersection point
    intersection = ray_origin.point + t * ray_direction

    #vector distance between intersection point and x1
    p = intersection - rect.x1

    #normalized projections along the edges
    u = np.dot(p, edge1) / np.dot(edge1, edge1)
    v = np.dot(p, edge2) / np.dot(edge2, edge2)
    
    if 0 <= u <= 1 and 0 <= v <= 1:
        return intersection
    
    return None
