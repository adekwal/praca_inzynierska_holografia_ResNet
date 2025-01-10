import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from gerchberg_saxton import rec_gs
from propagate import propagate_plane_wave

# Configuration
dataset_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\dane50_compressed.h5"
kulki_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\dane_testowe\kulki.h5"
experimental_data_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\dane_testowe\dane_eksperymentalne.h5"
model_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\ResNet\trained_models_tf_gpu_50epok\epoch_50_model_checkpoint.keras"

# User setup
verify_results_with_dataset_file = True # test with .h5 database
verify_results_with_experimental_data = False
image_index = 15

# Define simulation parameters
iter_max = 100
ph_init_mode = "null"
regularize = 1
constraint = "A"

nx = int(1024 / 2)
ny = int(1024 / 2)
px = 512  # padded image size
py = 512
n_disp = 512

dx = 2.4
dy = dx
n0 = 1
wavelength = 0.561
delta_z = 8.2222e3
z_sample_ccd1 = 3.5578e3
z_sample_ccd2 = z_sample_ccd1 + delta_z
delta_ph = np.pi / 2

# Set the proper path depending on the previous choice
if verify_results_with_dataset_file:
    h5_path = dataset_path
elif verify_results_with_experimental_data:
    # h5_path = experimental_data_path
    pass
else:
    h5_path = kulki_path

# Load data from the H5 file
with h5py.File(h5_path, 'r') as f:
    if verify_results_with_dataset_file:
        i1 = f['inputs'][image_index]
        i2 = f['targets'][image_index]
        ph0 = f['phase0'][image_index]
        ph1 = f['phase1'][image_index]
    elif verify_results_with_experimental_data:
        # data = f['OH'][...]
        # i1 = data[:, :, 0]
        # i2 = data[:, :, 4]
        pass
    else:
        i1 = f['i_ccd1'][:]
        i2 = f['i_ccd2'][:]
        ph0 = f['ph_obj'][:]

model = tf.keras.models.load_model(model_path) # load the trained ResNet model

# Generate predicted intensity from the model
if verify_results_with_dataset_file:
    i1 = np.squeeze(i1)
    i2 = np.squeeze(i2)
    ph0 = np.squeeze(ph0)
    ph1 = np.squeeze(ph1)
i2_predicted = model.predict(i1[np.newaxis, :, :, np.newaxis])[0, :, :, 0]
print("Image has been generated")
print("Please wait...")

# Prepare the input data for the GS method
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
data_ = {
    'ph1Init': np.zeros(i1.shape),
    'dx': dx,
    'dy': dy,
    'n0': n0,
    'lambda': wavelength,
    'z': [0, z_sample_ccd1, z_sample_ccd2],
    'A1': np.sqrt(i1),
    'A2': np.sqrt(i2)
}
options = {'max_iter': iter_max}

# Apply the GS algorithm
ph1_predicted, loss = rec_gs(data, options) # use ResNet then Gerchberg-Saxton algorithm
print("Phase reconstructed: 1/2")
ph1_, loss_ = rec_gs(data_, options) # use .h5 then Gerchberg-Saxton algorithm
print("Phase reconstructed: 2/2")


# Gabor method
# ph_obj = data['img']
ph_obj = ph0
ph_obj = (ph_obj - np.min(ph_obj)) / (np.max(ph_obj) - np.min(ph_obj)) * delta_ph
amp_obj = np.ones_like(ph_obj)

# ph_obj = np.pad(ph_obj, [((py - ph_obj.shape[0]) // 2,), ((px - ph_obj.shape[1]) // 2,)], mode='edge')
ph_obj = ph_obj - np.median(ph_obj)
u_obj = amp_obj * np.exp(1j * ph_obj)
u_obj = np.squeeze(u_obj) # analyze single image (change shape to (512, 512))

# Simulation of two defocused intensity measurements
u_ccd1 = propagate_plane_wave(u_obj, z_sample_ccd1, n0, wavelength, dx, dy)
u_ccd2 = propagate_plane_wave(u_obj, z_sample_ccd2, n0, wavelength, dx, dy)

# Cropping
u_obj = u_obj[py//2-ny//2:py//2+ny//2, px//2-nx//2:px//2+nx//2]
u_ccd1 = u_ccd1[py//2-ny//2:py//2+ny//2, px//2-nx//2:px//2+nx//2]
u_ccd2 = u_ccd2[py//2-ny//2:py//2+ny//2, px//2-nx//2:px//2+nx//2]

i_ccd1 = np.abs(u_ccd1)**2
i_ccd2 = np.abs(u_ccd2)**2

# # Adding padding
# pad_x = (int(py/2-ny/2), int(py/2-ny/2))
# pad_y = (int(px/2-nx/2), int(px/2-nx/2))
# i_ccd1 = np.pad(i_ccd1, (pad_x, pad_y), mode='edge')

# Gabor reconstruction in the first plane
u_rec_gabor_z1 = propagate_plane_wave(np.sqrt(i_ccd2), -delta_z, n0, wavelength, dx, dy)
ph_rec_gabor_z1 = np.angle(u_rec_gabor_z1 / np.mean(u_rec_gabor_z1))

# Display the results at the reconstruction plane
u_rec_gabor_z0 = propagate_plane_wave(u_rec_gabor_z1, -z_sample_ccd1, n0, wavelength, dx, dy)
# u_rec_gs_z0 = propagate_plane_wave(u_rec_gs_z1, -z_sample_ccd1, n0, wavelength, dx, dy)

# ph_obj = np.angle(u_obj / np.mean(u_obj))
ph_rec_gabor_z0 = np.angle(u_rec_gabor_z0 / np.mean(u_rec_gabor_z0))
# ph_rec_gs_z0 = np.angle(u_rec_gs_z0 / np.mean(u_rec_gs_z0))


# Gerchberg-Saxton method
u_rec_gs_z1_predicted = np.sqrt(i1) * np.exp(1j * ph1_predicted)
u_rec_gs_z1_ = np.sqrt(i1) * np.exp(1j * ph1_)

u_rec_gs_z0_predicted = propagate_plane_wave(u_rec_gs_z1_predicted, -z_sample_ccd1, n0, wavelength, dx, dy)
u_rec_gs_z0_ = propagate_plane_wave(u_rec_gs_z1_, -z_sample_ccd1, n0, wavelength, dx, dy)

ph_rec_gs_z0_predicted = np.angle(u_rec_gs_z0_predicted / np.mean(u_rec_gs_z0_predicted))
ph_rec_gs_z0_ = np.angle(u_rec_gs_z0_ / np.mean(u_rec_gs_z0_))


# Show intensity results
vmin_i = min(i2.min(), i2_predicted.min())
vmax_i = max(i2.max(), i2_predicted.max())

fig1, axs = plt.subplots(1, 3, figsize=(15, 8))
fig1.suptitle("Intensity Comparison (ResNet 50 epochs)", fontsize=16)

im1 = axs[0].imshow(i2, vmin=vmin_i, vmax=vmax_i)
axs[0].set_title("i2")

im2 = axs[1].imshow(i2_predicted, vmin=vmin_i, vmax=vmax_i)
axs[1].set_title("i2 (ResNet)")

im3 = axs[2].imshow(i2 - i2_predicted, vmin=vmin_i, vmax=vmax_i)
axs[2].set_title("i2 difference")

cbar = fig1.colorbar(im3, ax=axs, location='right', shrink=0.8)
cbar.set_label("Intensity Value")


# Show phase results
if verify_results_with_dataset_file:
    vmin_ph1 = min(ph1.min(), ph_rec_gabor_z1.min(), ph1_.min(), ph1_predicted.min())
    vmax_ph1 = max(ph1.max(), ph_rec_gabor_z1.min(), ph1_.max(), ph1_predicted.max())

    fig2, axs = plt.subplots(1, 4, figsize=(15, 8))
    fig2.suptitle("Phase Comparison (ResNet 50 epochs)", fontsize=16)

    im1 = axs[0].imshow(ph1, vmin=vmin_ph1, vmax=vmax_ph1)
    axs[0].set_title("ph1")

    im2 = axs[1].imshow(ph_rec_gabor_z1, vmin=vmin_ph1, vmax=vmax_ph1)
    axs[1].set_title("ph1 (Gabor)")

    im3 = axs[2].imshow(ph1_, vmin=vmin_ph1, vmax=vmax_ph1)
    axs[2].set_title("ph1 (GS)")

    im4 = axs[3].imshow(ph1_predicted, vmin=vmin_ph1, vmax=vmax_ph1)
    axs[3].set_title("ph1 (GS + ResNet)")

    cbar = fig2.colorbar(im4, ax=axs, location='right', shrink=0.8)
    cbar.set_label("Phase Value")


# Show phase at object plane results
vmin_ph0 = min(ph0.min(), ph_rec_gabor_z0.min(), ph_rec_gs_z0_.min(), ph_rec_gs_z0_predicted.min())
vmax_ph0 = max(ph0.max(), ph_rec_gabor_z0.max(), ph_rec_gs_z0_.max(), ph_rec_gs_z0_predicted.max())

fig3, axs = plt.subplots(1, 4, figsize=(15, 8))
fig3.suptitle("Phase Comparison (ResNet 50 epochs) at object plane", fontsize=16)

im1 = axs[0].imshow(ph0, vmin=vmin_ph0, vmax=vmax_ph0)
axs[0].set_title("ph0")

im2 = axs[1].imshow(ph_rec_gabor_z0, vmin=vmin_ph0, vmax=vmax_ph0)
axs[1].set_title("ph0 (Gabor)")

im3 = axs[2].imshow(ph_rec_gs_z0_, vmin=vmin_ph0, vmax=vmax_ph0)
axs[2].set_title("ph0 (GS)")

im4 = axs[3].imshow(ph_rec_gs_z0_predicted, vmin=vmin_ph0, vmax=vmax_ph0)
axs[3].set_title("ph0 (GS + ResNet)")

cbar = fig3.colorbar(im4, ax=axs, location='right', shrink=0.8)
cbar.set_label("Phase Value")

plt.show()

# Display cost function
plt.figure()
plt.plot(loss)
plt.title('Optimization progress')
plt.xlabel('Iteration number')
plt.ylabel('Log(cost)')
plt.show()