import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gerchberg_saxton import rec_gs
from propagate import propagate_plane_wave

# Configuration
h5_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\dane50_compressed.h5"
model_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\ResNet\trained_models_tf_gpu_50epok\epoch_50_model_checkpoint.keras"
# image_index = 0

# Define simulation parameters
iter_max = 100
ph_init_mode = "null"
regularize = 1
constraint = "A"

px = 512
py = 512
nx = 512
ny = 512
n_disp = 512

dx = 2.4
dy = dx
n0 = 1
wavelength = 0.561
delta_z = 8.2222e3
z_sample_ccd1 = 3.5578e3
z_sample_ccd2 = z_sample_ccd1 + delta_z
delta_ph = np.pi / 2

# Load data
data = np.load("obraz.npz") # load image as npz
i1 = data[data.files[0]]
i1 = np.squeeze(i1) # single image

# Load the trained Keras model
model = tf.keras.models.load_model(model_path)

# Generate i2_predicted from the model
i2_predicted = model.predict(i1[np.newaxis, :, :, np.newaxis])[0, :, :, 0]
i2_predicted = np.squeeze(i2_predicted) # single image
print("Image has been generated")
print("Please wait...")

ph_obj = data['img']
ph_obj = (ph_obj - np.min(ph_obj)) / (np.max(ph_obj) - np.min(ph_obj)) * delta_ph
amp_obj = np.ones_like(ph_obj)

# ph_obj = np.pad(ph_obj, [((py - ph_obj.shape[0]) // 2,), ((px - ph_obj.shape[1]) // 2,)], mode='edge')
ph_obj = ph_obj - np.median(ph_obj)
u_obj = amp_obj * np.exp(1j * ph_obj)
u_obj = np.squeeze(u_obj) # single image

# Simulation of two defocused intensity measurements
u_ccd1 = propagate_plane_wave(u_obj, z_sample_ccd1, n0, wavelength, dx, dy)
u_ccd2 = propagate_plane_wave(u_obj, z_sample_ccd2, n0, wavelength, dx, dy)

# Cropping
u_obj = u_obj[py//2-ny//2:py//2+ny//2, px//2-nx//2:px//2+nx//2]
u_ccd1 = u_ccd1[py//2-ny//2:py//2+ny//2, px//2-nx//2:px//2+nx//2]
u_ccd2 = u_ccd2[py//2-ny//2:py//2+ny//2, px//2-nx//2:px//2+nx//2]

i_ccd1 = np.abs(u_ccd1)**2
i_ccd2 = np.abs(u_ccd2)**2

# Gerchberg-Saxton reconstruction method
data = {
    'ph1Init': np.zeros(i1.shape),
    'dx': dx,
    'dy': dy,
    'n0': n0,
    'lambda': wavelength,
    'z': [0, z_sample_ccd1, z_sample_ccd2],
    'A1': np.sqrt(i1),
    'A2': np.sqrt(i2_predicted)
}
options = {'max_iter': iter_max}

ph_rec_gs_z1, loss = rec_gs(data, options)
u_rec_gs_z1 = np.sqrt(i_ccd1) * np.exp(1j * ph_rec_gs_z1)

u_rec_gs_z0 = propagate_plane_wave(u_rec_gs_z1, -z_sample_ccd1, n0, wavelength, dx, dy)

ph_obj = np.angle(u_obj / np.mean(u_obj))
ph_rec_gs_z0 = np.angle(u_rec_gs_z0 / np.mean(u_rec_gs_z0))
ph_z0_range = [np.min(ph_rec_gs_z0), np.max(ph_rec_gs_z0)]

# Show phase results
vmin_ph = min(ph_obj.min(), ph_rec_gs_z0.min(), (ph_rec_gs_z0 - ph_obj).min())
vmax_ph = max(ph_obj.max(), ph_rec_gs_z0.max(), (ph_rec_gs_z0 - ph_obj).max())

fig1, axs = plt.subplots(1, 3, figsize=(15, 8))
fig1.suptitle("phase comparison (ResNet 50 epochs)", fontsize=16)

im1 = axs[0].imshow(ph_obj, vmin=vmin_ph, vmax=vmax_ph)
axs[0].set_title("Ground truth phase - object plane")

im2 = axs[1].imshow(ph_rec_gs_z0, vmin=vmin_ph, vmax=vmax_ph)
axs[1].set_title("GS phase - object plane")

im3 = axs[2].imshow(ph_rec_gs_z0 - ph_obj, vmin=vmin_ph, vmax=vmax_ph)
axs[2].set_title("Difference")

cbar = fig1.colorbar(im3, ax=axs, location='right', shrink=0.8)
cbar.set_label("Phase Value")

plt.show()

# Display cost function
plt.figure()
plt.plot(loss)
plt.title('Optimization progress')
plt.xlabel('Iteration number')
plt.ylabel('Log(cost)')
plt.show()