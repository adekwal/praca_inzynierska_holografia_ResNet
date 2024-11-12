import numpy as np
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from GenerateData import generate_data

# Define simulation parameters
data_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\flowers"
training_dirs = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
nx = int(1024 / 2)
ny = int(1024 / 2)
px = 1024  # padded image size
py = 1024
dx = 2.4
dy = dx
n0 = 1
lambda_ = 0.1
delta_z = 8e3
z1 = 3.5e3
z2 = z1 + delta_z
delta_ph_max = np.pi / 2
img_count = 1000
n = 10  # number of images per subfolder to process

# Checking propagation condition for angular spectrum
print(lambda_ * delta_z < min([px, py]) * dx * dx)
print(lambda_ * z2 < min([px, py]) * dx * dx)

# Initial configurations
inputs = np.zeros((ny, nx, 1, img_count))
targets = np.zeros((ny, nx, 1, img_count))
phase1 = np.zeros((ny, nx, 1, img_count))
phase0 = np.zeros((ny, nx, 1, img_count))

# Generate GS data
img_total_no = 0

for i_dir in training_dirs:
    folder_path = os.path.join(data_path, i_dir)
    img_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    img_files = img_files[:n]  # limit to `n` images per folder

    for i_img, img_file in enumerate(img_files):
        img_total_no += 1
        if img_total_no > img_count:
            break

        # Load image
        img = imageio.imread(os.path.join(folder_path, img_file)).astype(float)
        i1, i2, ph1, ph2, ph0 = generate_data(img, delta_ph_max, z1, z2, lambda_, dx, nx, ny, px, py)

        # Cropping and storing data
        i1 = i1[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]
        i2 = i2[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]
        ph1 = ph1[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]
        ph2 = ph2[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]
        ph0 = ph0[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]

        inputs[:, :, 0, img_total_no-1] = i1
        targets[:, :, 0, img_total_no-1] = i2
        phase1[:, :, 0, img_total_no-1] = ph1
        phase0[:, :, 0, img_total_no-1] = ph0

        ph0 -= np.median(ph0)
        ph1 -= np.median(ph1)
        ph2 -= np.median(ph2)

        print(img_total_no)

# Save data
np.savez('training_data.npz', inputs=inputs, targets=targets, phase0=phase0, phase1=phase1)
print("done")