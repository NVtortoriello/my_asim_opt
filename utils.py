import numpy as np


def point_distance(point1, point2):
    return np.linalg.norm(point2-point1,2)

    
def unit_vector(point1, point2):
    return (point2 - point1) / np.linalg.norm(point2 - point1,2)


def poly_length(points):

    length = 0

    for k in range(len(points)-1):
        length += point_distance(points[k], points[k+1])

    return length


def s_to_c(theta, phi):

    r = np.zeros((3,3))

    r[0,0] = np.sin(theta) * np.cos(phi)
    r[0,1] = np.cos(theta) * np.cos(phi)
    r[0,2] = -np.sin(phi)
    r[1,0] = np.sin(theta) * np.sin(phi)
    r[1,1] = np.cos(theta) * np.sin(phi)
    r[1,2] = np.cos(phi)
    r[2,0] = np.cos(theta)
    r[2,1] = -np.sin(theta)
    r[2,2] = 0

    return r


def rect_normal(rect):

    edge1 = rect.x2 - rect.x1
    edge2 = rect.x3 - rect.x1
    
    normal = np.cross(edge1, edge2)
    normal = normal / np.linalg.norm(normal)

    return normal