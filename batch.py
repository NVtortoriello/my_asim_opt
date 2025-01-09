import numpy as np

import torch
import torch.optim as optim
import time

from transforms import transform_l_to_g
from utils import unit_vector, point_distance
from utd import r_dyad, torch_r_dyad 


def generate_batch(tx, rfs, rxs, normal, eps, sigma):
    #batch of rays
    batch = []
    #FF global coordinates
    field = np.sqrt(tx.power) * np.array([complex(0,1), complex(0,0)])

    for idx, rx in enumerate(rxs):

        rf = rfs[idx]

        inc, dep, trs, m_r = propagate(tx, rf, rx, normal, field, eps, sigma)

        batch.append((inc, dep, trs, m_r))

    return batch


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
    

    m_r, trs = r_dyad(inc, dep, normal, er)
    #m_r = m_r + t_dyad(inc, dep, normal, er)

    # sp_factor = 1 / pst_radius

    # ifield = (wl / 4 / np.pi) * m_r @ ifield * sp_factor

    # ifield = transform_g_to_l(ifield, tx.b_az, tx.b_zt, 0, unit_vector(rf, rx.point))
    # ifield *= np.array([complex(0,1), complex(0,0)])

    # delay = poly_length([tx.point, rf, rx.point]) / c

    return inc, dep, trs, m_r


def complex_mse_loss(output, target):
    return torch.norm(((output - target)**2).mean(dtype=torch.complex64))


def train(tx, rfs, rxs, normal, eps, sigma):

    batch = generate_batch(tx, rfs, rxs, normal, eps, sigma)

    incs_np = np.array([x[0] for x in batch])
    deps_np = np.array([x[1] for x in batch])
    trss_np = np.array([x[2] for x in batch])

    #per ogni punto di interesse definisce una matrice 
    m_rs_np = np.array([x[3] for x in batch])

    nors_np = np.zeros(incs_np.shape)

    nors_np[..., :] = normal
    
    incs = torch.from_numpy(incs_np)                #incident  rays batch
    deps = torch.from_numpy(deps_np)                #reflected rays batch
    trss = torch.from_numpy(trss_np)                #reflected rays batch

    m_rs = torch.from_numpy(m_rs_np)                #conversion matrixs batch---TARGET/GND TRUTH
    nors = torch.from_numpy(nors_np)

    model = torch_r_dyad()
    criterion = complex_mse_loss  # Mean squared error loss
    optimizer = optim.Adam(model.parameters(), lr=0.1)  # Stochastic gradient descent

    print(f"actual parameters: eps = {float(eps)}, sigma = {sigma}")

    # Training loop

    start = time.time()

    num_epochs = 180
    for epoch in range(num_epochs):
        # Forward pass: process of passing input data through the network to compute the output.
        predictions = model((incs, deps, nors, trss))
        loss = criterion(predictions, m_rs)

        # Backward pass
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        # Print progress every 100 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, eps: {model.eps.item():.4f}, sigma: {model.conductivity.item():.4f}")

    end = time.time()
    # Final parameters
    print(f"learned parameters: eps = {np.round(model.eps.item(),3)}, sigma = {np.round(model.conductivity.item(),3)}")

    print(f'runtime: {np.round(end-start,3)} sec')