import numpy as np

import torch
import torch.nn as nn

from transforms import transform_g_to_l, transform_l_to_g
from utils import unit_vector, rect_normal, point_distance, poly_length
from utd import r_dyad, torch_r_dyad


def generate_batch(tx, rfs, rxs, normal, eps, sigma):

    rays = []

    field = np.sqrt(tx.power) * np.array([complex(0,1), complex(0,0)])

    for idx, rx in enumerate(rxs):

        rf = rfs[idx]

        delay, ifield = propagate(tx, rf, rx, normal, field, eps, sigma)

        rays.append((delay, ifield))

    return rays


def propagate(tx, rf, rx, normal, field, eps, sigma):

    c = 299792458
    f = 3.6e9
    wl = c / f
    wv = 2 * np.pi / wl

    e0 = 8.8541878188e-12

    conductivity = sigma
    er = complex(eps, - conductivity / 2 / np.pi / f / e0)  

    ifield = transform_l_to_g(field, tx.b_az, tx.b_zt, 0, unit_vector(tx.point, rf))

    pre_radius = point_distance(tx.point, rf)
    delta = point_distance(rf, rx.point)
    pst_radius = pre_radius + delta

    ifield *= np.exp(complex(0, -wv*pst_radius)) / pst_radius

    inc = unit_vector(tx.point, rf)
    dep = unit_vector(rf, rx.point)

    m_r = r_dyad(inc, dep, normal, er)

    sp_factor = 1 / pst_radius

    ifield = (wl / 4 / np.pi) * m_r @ ifield * sp_factor

    ifield = transform_g_to_l(ifield, tx.b_az, tx.b_zt, 0, unit_vector(rf, rx.point))
    ifield *= np.array([complex(0,1), complex(0,0)])

    delay = poly_length([tx.point, rf, rx.point]) / c

    return delay, ifield


class torch_propagate(nn.module):

    def __init__(self, normal):
        super(torch_propagate, self).__init__()
        # Initialize parameters a and b as learnable variables
        
        self.c = 299792458
        self.f = 3.6e9
        self.wl = self.c / self.f
        self.wv = 2 * np.pi / self.wl
        self.e0 = 8.8541878188e-12

        self.normal = normal
        
        self.conductivity = nn.Parameter(torch.tensor(1.0))
        self.eps = nn.Parameter(torch.tensor(1.0))

        self.r_dyad = torch_r_dyad(self.normal)
        
    def forward(self, tx, rf, rx, field):

        ifield = transform_l_to_g(field, tx.b_az, tx.b_zt, 0, unit_vector(tx.point, rf))

        pre_radius = point_distance(tx.point, rf)
        delta = point_distance(rf, rx.point)
        pst_radius = pre_radius + delta

        ifield *= np.exp(complex(0, -self.wv*pst_radius)) / pst_radius

        inc = unit_vector(tx.point, rf)
        dep = unit_vector(rf, rx.point)

        er = torch.complex(self.eps, -self.conductivity / 2 / np.pi / self.e0)

        m_r = self.r_dyad.forward(inc, dep, er)

        sp_factor = 1 / pst_radius

        ifield = (self.wl / 4 / np.pi) * m_r @ ifield * sp_factor

        ifield = transform_g_to_l(ifield, tx.b_az, tx.b_zt, 0, unit_vector(rf, rx.point))
        ifield *= np.array([complex(0,1), complex(0,0)])

        delay = poly_length([tx.point, rf, rx.point]) / self.c

        return delay, ifield
