import numpy as np
import propagate

def rec_gs(data, opts):
    ph_init = data['ph1Init']
    dx = data['dx']
    dy = data['dy']
    n0 = data['n0']
    lambda_ = data['lambda']
    z_vec = data['z']
    amp1 = data['A1']
    amp2 = data['A2']

    u_rec_z1 = amp1 * np.exp(1j * ph_init)
    loss = np.zeros(opts['max_iter'])

    for iter in range(opts['max_iter']):
        u_rec_z2 = propagate.propagate_plane_wave(u_rec_z1, z_vec[2] - z_vec[1], n0, lambda_, dx, dy)
        loss[iter] = (1/4) * np.sum((amp2**2 - np.abs(u_rec_z2)**2)**2)
        u_rec_z2 = amp2 * np.exp(1j * np.angle(u_rec_z2))
        u_rec_z1 = propagate.propagate_plane_wave(u_rec_z2, z_vec[1] - z_vec[2], n0, lambda_, dx, dy)
        u_rec_z1 = amp1 * np.exp(1j * np.angle(u_rec_z1))

    ph_rec_z1 = np.angle(u_rec_z1)

    return ph_rec_z1, loss
