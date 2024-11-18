import numpy as np
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from training_data_gen import generate_data

## Define simulation parameters
data_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\flowers"
training_dirs = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
nx = int(1024 / 2)
ny = int(1024 / 2)
px = 1024  # padded image size
py = 1024
dx = 2.4
dy = dx
n0 = 1
# lambda_ = 0.561
lambda_ = 0.1
delta_z = 8e3
z1 = 3.5e3
z2 = z1 + delta_z
delta_ph_max = np.pi / 2
img_count = 1000

## Define user setup
do_you_want_to_print_your_result = False # change it if you want to show images

## Checking propagation condition for angular spectrum
print(lambda_ * delta_z < min([px, py]) * dx * dx)
print(lambda_ * z2 < min([px, py]) * dx * dx)


## Initial configurations
inputs = np.zeros((ny, nx, 1, img_count))
targets = np.zeros((ny, nx, 1, img_count))
phase1 = np.zeros((ny, nx, 1, img_count))
phase0 = np.zeros((ny, nx, 1, img_count))


## Generate GS data
img_total_no = 0
directories = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

for i_dir in directories:
    img_files = [f for f in os.listdir(os.path.join(data_path, i_dir)) if
                 os.path.isfile(os.path.join(data_path, i_dir, f))]

    for i_img in range(0, int(np.ceil(img_count / len(directories)))):
        img_total_no += 1
        if img_total_no > img_count:
            break

        # load image
        img = imageio.imread(os.path.join(data_path, i_dir, img_files[i_img])).astype(float)
        i1, i2, ph1, ph0 = generate_data(img, delta_ph_max, z1, z2, lambda_, dx, nx, ny, px, py)

        i1 = i1[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]
        i2 = i2[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]
        ph1 = ph1[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]
        ph0 = ph0[int(py / 2 - ny / 2): int(py / 2 + ny / 2), int(px / 2 - nx / 2): int(px / 2 + nx / 2)]

        inputs[:, :, 0, img_total_no-1] = i1
        targets[:, :, 0, img_total_no-1] = i2
        phase1[:, :, 0, img_total_no-1] = ph1
        phase0[:, :, 0, img_total_no-1] = ph0

        ph0 -= np.median(ph0)
        ph1 -= np.median(ph1)

        i_range = [np.min(i1), np.max(i1)]
        ph_range = [np.min(ph0), np.max(ph0)]

        if do_you_want_to_print_your_result:
            plt.figure(1, figsize=(10, 6))
            plt.clf()
            plt.suptitle(f'Image #{img_total_no}')

            plt.subplot(2, 2, 1)
            plt.imshow(i1, vmin=i_range[0], vmax=i_range[1], cmap='hot')
            plt.title('i1')
            plt.axis('image')

            plt.subplot(2, 2, 2)
            plt.imshow(i2, vmin=i_range[0], vmax=i_range[1], cmap='hot')
            plt.title('i2')
            plt.axis('image')

            plt.subplot(2, 2, 3)
            plt.imshow(ph0, vmin=ph_range[0], vmax=ph_range[1], cmap='spring')
            plt.title('ph0')
            plt.axis('image')

            plt.subplot(2, 2, 4)
            plt.imshow(ph1, vmin=ph_range[0], vmax=ph_range[1], cmap='spring')
            plt.title('ph1')
            plt.axis('image')

            plt.pause(1)

        print(img_total_no)

np.savez('training_data.npz', inputs=inputs, targets=targets, phase0=phase0, phase1=phase1)
print("\ndone")