import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from gerchberg_saxton import rec_gs
from propagate import propagate_plane_wave
from plots import plot_charts
from prepare_image import resize
from prepare_image import crop

# Configuration
h5_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\dane50_compressed.h5"
# h5_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\dane_testowe\kulki.h5"
# h5_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\dane_testowe\dane_eksperymentalne.h5"
model_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\ResNet\trained_models_tf_gpu_50epok\epoch_50_model_checkpoint.keras"

# User setup
use_training_set = True # set True if using training set because of different label
image_index = 0

# Define simulation parameters
iter_max = 200
ph_init_mode = "null"
regularize = 1
constraint = "A"

nx = int(1024 / 2)
ny = int(1024 / 2)
px = 1024 # padded image size
py = 1024
x_pos = 500
y_pos = 1800

dx = 2.4
dy = dx
n0 = 1
wavelength = 0.561
delta_z = 8.2222e3
z_sample_ccd1 = 3.5578e3
z_sample_ccd2 = z_sample_ccd1 + delta_z
delta_ph = np.pi / 2


# Load data from the H5 file
with h5py.File(h5_path, 'r') as f:
    if use_training_set:
        i1 = f['inputs'][image_index]
        i2 = f['targets'][image_index]
        # ph0 = f['phase0'][image_index]
        # ph1 = f['phase1'][image_index]
    else:
        i1 = f['i_ccd1'][:]
        i2 = f['i_ccd2'][:]

if not use_training_set:
    if i1.shape != (512, 512):
        i1 = resize(i1, nx, ny, px, py, x_pos, y_pos)
        i2 = resize(i2, nx, ny, px, py, x_pos, y_pos)


# Generate predicted intensity from the model
i1 = np.squeeze(i1)
i2 = np.squeeze(i2)
model = tf.keras.models.load_model(model_path) # load the trained ResNet model
i2_predicted = model.predict(i1[np.newaxis, :, :, np.newaxis])[0, :, :, 0]
print("Image has been generated")
print("Please wait...")

# Remove and duplicate 10 px at edges
i2_predicted = i2_predicted[10:-10, 10:-10]
i2_predicted = np.pad(i2_predicted, pad_width=10, mode='edge')

# Padding to 1024x1024
pad_x = (int(py / 2 - ny / 2), int(py / 2 - ny / 2))
pad_y = (int(px / 2 - nx / 2), int(px / 2 - nx / 2))
i1 = np.pad(i1, (pad_x, pad_y), mode='edge')
i2 = np.pad(i2, (pad_x, pad_y), mode='edge')
i2_predicted = np.pad(i2_predicted, (pad_x, pad_y), mode='edge')


def create_data(network_data):
    return {
        'ph1Init': np.zeros(i1.shape),
        'dx': dx,
        'dy': dy,
        'n0': n0,
        'lambda': wavelength,
        'z': [0, z_sample_ccd1, z_sample_ccd2],
        'A1': np.sqrt(i1),
        'A2': np.sqrt(i2) if not network_data else np.sqrt(i2_predicted)
    }
options = {'max_iter': iter_max}


## Gabor method
i_ccd2 = i2

# Gabor reconstruction in the first plane
u_rec_gabor_z1 = propagate_plane_wave(np.sqrt(i_ccd2), -delta_z, n0, wavelength, dx, dy)
ph_rec_gabor_z1 = np.angle(u_rec_gabor_z1 / np.mean(u_rec_gabor_z1))

# Gabor reconstruction in the object plane
u_rec_gabor_z0 = propagate_plane_wave(u_rec_gabor_z1, -z_sample_ccd1, n0, wavelength, dx, dy)
ph_rec_gabor_z0 = np.angle(u_rec_gabor_z0 / np.mean(u_rec_gabor_z0))


## Gerchberg-Saxton method
ph1_predicted, loss = rec_gs(create_data(True), options) # use ResNet then Gerchberg-Saxton algorithm
print("Phase reconstructed: 1/2")
ph1_perfect, _ = rec_gs(create_data(False), options) # use .h5 then Gerchberg-Saxton algorithm
print("Phase reconstructed: 2/2")

u_rec_gs_z1_predicted = np.sqrt(i1) * np.exp(1j * ph1_predicted)
u_rec_gs_z1_perfect = np.sqrt(i1) * np.exp(1j * ph1_perfect)

# Back propagation
u_rec_gs_z0_predicted = propagate_plane_wave(u_rec_gs_z1_predicted, -z_sample_ccd1, n0, wavelength, dx, dy)
u_rec_gs_z0_perfect = propagate_plane_wave(u_rec_gs_z1_perfect, -z_sample_ccd1, n0, wavelength, dx, dy)

# Calculate phase at object plane
ph_rec_gs_z0_predicted = np.angle(u_rec_gs_z0_predicted / np.mean(u_rec_gs_z0_predicted))
ph_rec_gs_z0_perfect = np.angle(u_rec_gs_z0_perfect / np.mean(u_rec_gs_z0_perfect))


# Cropping to 512x512
i1 = crop(i1, nx, ny, px, py)
i2 = crop(i2, nx, ny, px, py)
i2_predicted = crop(i2_predicted, nx, ny, px, py)

ph_rec_gabor_z1 = crop(ph_rec_gabor_z1, nx, ny, px, py)
ph1_perfect = crop(ph1_perfect, nx, ny, px, py)
ph1_predicted = crop(ph1_predicted, nx, ny, px, py)

ph_rec_gabor_z0 = crop(ph_rec_gabor_z0, nx, ny, px, py)
ph_rec_gs_z0_perfect = crop(ph_rec_gs_z0_perfect, nx, ny, px, py)
ph_rec_gs_z0_predicted = crop(ph_rec_gs_z0_predicted, nx, ny, px, py)


## Show results
i_charts = [i1, i2, i2_predicted]
i_titles = ["i1", "i2", "i2 (ResNet)"]
plot_charts(i_charts, i_titles, suptitle="Intensity", cbar_label="Intensity Value")

ph1_charts = [ph_rec_gabor_z1, ph1_perfect, ph1_predicted]
ph1_titles = ["ph1 (Gabor)", "ph1 (GS)", "ph1 (GS + ResNet)"]
plot_charts(ph1_charts, ph1_titles, suptitle="Phase ph1 comparison", cbar_label="Phase Value")

ph0_charts = [ph_rec_gabor_z0, ph_rec_gs_z0_perfect, ph_rec_gs_z0_predicted]
ph0_titles = ["ph0 (Gabor)", "ph0 (GS)", "ph0 (GS + ResNet)"]
plot_charts(ph0_charts, ph0_titles, suptitle="Phase ph0 comparison", cbar_label="Phase Value")

# Show cost function
plt.figure()
plt.plot(loss)
plt.title('Optimization progress')
plt.xlabel('Iteration number')
plt.ylabel('Log(cost)')
plt.show()
