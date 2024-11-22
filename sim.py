import numpy as np
from copy import deepcopy

class sim_wall:

    def __init__(self, x1, x3):

        self.x1 = deepcopy(x1)  # bottom left
        self.x2 = deepcopy(x3)
        self.x3 = deepcopy(x3)
        self.x4 = deepcopy(x1)
        
        self.x2[2] = self.x1[2]
        self.x4[2] = self.x3[2]

        self.eps = 5
        self.sigma = 0.1

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
    
    # Define two edges of the rectangle
    edge1 = rect.x2 - rect.x1
    edge2 = rect.x3 - rect.x1
    
    # Compute the normal of the rectangle's plane
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)  # Normalize the normal
    
    # Find the projection of the point onto the plane
    rect_point = rect.x1  # Any point on the rectangle will do
    to_point = ap.point - rect_point
    distance_to_plane = np.dot(to_point, normal)
    projection = ap.point - distance_to_plane * normal
    
    # Compute the mirrored point
    mirrored_point = projection - (ap.point - projection)
    
    return mirrored_point


def ray_intersection(ray_origin, ray_direction, rect):
    
    # Define two edges of the rectangle
    edge1 = rect.x2 - rect.x1
    edge2 = rect.x3 - rect.x1
    
    # Compute the normal of the rectangle's plane
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)  # Normalize the normal
    
    # Check if the ray is parallel to the plane (dot product with normal is zero)
    denom = np.dot(normal, ray_direction)
    if np.isclose(denom, 0):
        return None  # No intersection, ray is parallel to the plane
    
    # Calculate intersection with the plane
    d = np.dot(normal, rect.x1)
    t = (d - np.dot(normal, ray_origin.point)) / denom
    
    if t < 0:
        return None  # Intersection is behind the ray origin
    
    # Compute the intersection point
    intersection = ray_origin.point + t * ray_direction
    
    # Check if the intersection point lies within the rectangle
    # Use the parametric equations of the rectangle
    p = intersection - rect.x1
    u = np.dot(p, edge1) / np.dot(edge1, edge1)
    v = np.dot(p, edge2) / np.dot(edge2, edge2)
    
    if 0 <= u <= 1 and 0 <= v <= 1:
        return intersection  # Intersection point is within the rectangle
    
    return None  # Intersection point is outside the rectangle
