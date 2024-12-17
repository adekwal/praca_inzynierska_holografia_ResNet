import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import propagate
import gerchberg_saxton as gs


# Define simulation parameters
iter_max = 100
ph_init_mode = "null"
regularize = 1
constraint = "A"

px = 1024
py = 1024
nx = 1024
ny = 1024
n_disp = 512

dx = 2.4
dy = dx
n0 = 1
wavelength = 0.561
delta_z = 8e3
z_sample_ccd1 = 3.5e3
z_sample_ccd2 = z_sample_ccd1 + delta_z
delta_ph = np.pi / 4

# # Generation of object wave
# amp_obj = np.ones((py, px))
# data = loadmat("my_coin.mat")
# ph_obj = data['img']
# ph_obj = (ph_obj - np.min(ph_obj)) / (np.max(ph_obj) - np.min(ph_obj)) * delta_ph

data = np.load("obraz.npz")
ph_obj = data['img']
ph_obj = (ph_obj - np.min(ph_obj)) / (np.max(ph_obj) - np.min(ph_obj)) * delta_ph
amp_obj = np.ones_like(ph_obj)

ph_obj = np.pad(ph_obj, [((py - ph_obj.shape[0]) // 2,), ((px - ph_obj.shape[1]) // 2,)], mode='edge')
ph_obj = ph_obj - np.median(ph_obj)
u_obj = amp_obj * np.exp(1j * ph_obj)

# Simulation of two defocused intensity measurements
u_ccd1 = propagate.propagate_plane_wave(u_obj, z_sample_ccd1, n0, wavelength, dx, dy)
u_ccd2 = propagate.propagate_plane_wave(u_obj, z_sample_ccd2, n0, wavelength, dx, dy)

# Cropping
u_obj = u_obj[py//2-ny//2:py//2+ny//2, px//2-nx//2:px//2+nx//2]
u_ccd1 = u_ccd1[py//2-ny//2:py//2+ny//2, px//2-nx//2:px//2+nx//2]
u_ccd2 = u_ccd2[py//2-ny//2:py//2+ny//2, px//2-nx//2:px//2+nx//2]

i_ccd1 = np.abs(u_ccd1)**2
i_ccd2 = np.abs(u_ccd2)**2

# Display intensity measurements
i_ccd1_range = [np.min(i_ccd1), np.max(i_ccd1)]
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(i_ccd1, vmin=i_ccd1_range[0], vmax=i_ccd1_range[1], cmap='gray')
plt.title("Intensity at 1st detection plane")
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(i_ccd2, vmin=i_ccd1_range[0], vmax=i_ccd1_range[1], cmap='gray')
plt.title("Intensity at 2nd detection plane")
plt.colorbar()
plt.show()

# Gabor reconstruction in the first plane
u_rec_gabor_z1 = propagate.propagate_plane_wave(np.sqrt(i_ccd2), -delta_z, n0, wavelength, dx, dy)

# Gerchberg-Saxton reconstruction method
data = {
    'ph1Init': np.zeros_like(u_rec_gabor_z1),
    'dx': dx,
    'dy': dy,
    'n0': n0,
    'lambda': wavelength,
    'z': [0, z_sample_ccd1, z_sample_ccd2],
    'A1': np.sqrt(i_ccd1),
    'A2': np.sqrt(i_ccd2),
}
options = {'max_iter': iter_max}

ph_rec_gs_z1, loss = gs.rec_gs(data, options)
u_rec_gs_z1 = np.sqrt(i_ccd1) * np.exp(1j * ph_rec_gs_z1)

# Display the results at the reconstruction plane
u_rec_gabor_z0 = propagate.propagate_plane_wave(u_rec_gabor_z1, -z_sample_ccd1, n0, wavelength, dx, dy)
u_rec_gs_z0 = propagate.propagate_plane_wave(u_rec_gs_z1, -z_sample_ccd1, n0, wavelength, dx, dy)

ph_obj = np.angle(u_obj / np.mean(u_obj))
ph_rec_gabor_z0 = np.angle(u_rec_gabor_z0 / np.mean(u_rec_gabor_z0))
ph_rec_gs_z0 = np.angle(u_rec_gs_z0 / np.mean(u_rec_gs_z0))

ph_z0_range = [np.min(ph_rec_gs_z0), np.max(ph_rec_gs_z0)]

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(ph_obj, vmin=ph_z0_range[0], vmax=ph_z0_range[1], cmap='gray')
plt.title("Ground truth phase - object plane")
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(ph_rec_gabor_z0, vmin=ph_z0_range[0], vmax=ph_z0_range[1], cmap='gray')
plt.title("GS phase - object plane")
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(ph_rec_gabor_z0 - ph_obj, cmap='gray')
plt.title("Difference")
plt.colorbar()
plt.show()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(ph_obj, vmin=ph_z0_range[0], vmax=ph_z0_range[1], cmap='gray')
plt.title("Ground truth phase - object plane")
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(ph_rec_gs_z0, vmin=ph_z0_range[0], vmax=ph_z0_range[1], cmap='gray')
plt.title("GS phase - object plane")
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(ph_rec_gs_z0 - ph_obj, cmap='gray')
plt.title("Difference")
plt.colorbar()
plt.show()

# Display cost function
plt.figure()
plt.plot(loss)
plt.title('Optimization progress')
plt.xlabel('Iteration number')
plt.ylabel('Log(cost)')
plt.show()

