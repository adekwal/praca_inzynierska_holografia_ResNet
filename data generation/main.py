import numpy as np
import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import h5py
from training_data_gen import generate_data

# Define simulation parameters
data_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\flowers"
nx = int(1024 / 2)
ny = int(1024 / 2)
px = 1024  # padded image size
py = 1024
dx = 2.4
dy = dx
n0 = 1
lambda_ = 0.561
delta_z = 8.2222e3
z1 = 3.5578e3
z2 = z1 + delta_z
delta_ph_max = np.pi / 2

# Define user setup
output_file = 'dane'  # define the filename
img_count = 50  # choose the number of images you want to process
save_as_npz = False
save_as_h5 = True
save_disc_space = False  # compress data file
show_images = False  # show generated data

# Checking propagation condition for angular spectrum
print(lambda_ * delta_z < min([px, py]) * dx * dx)
print(lambda_ * z2 < min([px, py]) * dx * dx)
print()

# Initialize containers for NPZ
if save_as_npz:
    inputs = []
    targets = []
    phase0 = []
    phase1 = []

# HDF5 setup
if save_as_h5:
    h5_file = f'{output_file}.h5'

    with h5py.File(h5_file, 'w') as h5f:
        h5f.create_dataset('inputs', shape=(0, nx, ny, 1), maxshape=(None, nx, ny, 1), compression=None,
                           dtype='float32')
        h5f.create_dataset('targets', shape=(0, nx, ny, 1), maxshape=(None, nx, ny, 1), compression=None,
                           dtype='float32')
        h5f.create_dataset('phase0', shape=(0, nx, ny, 1), maxshape=(None, nx, ny, 1), compression=None,
                           dtype='float32')
        h5f.create_dataset('phase1', shape=(0, nx, ny, 1), maxshape=(None, nx, ny, 1), compression=None,
                           dtype='float32')

# Process each image
img_files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
img_files = img_files[:img_count]

for img_no, img_file in enumerate(img_files, start=1):
    img = imageio.imread(os.path.join(data_path, img_file)).astype(float)
    i1, i2, ph1, ph0 = generate_data(img, delta_ph_max, z1, z2, lambda_, dx, nx, ny, px, py)  # generate data

    # Save data to NPZ lists
    if save_as_npz:
        inputs.append(i1[..., np.newaxis])
        targets.append(i2[..., np.newaxis])
        phase0.append(ph0[..., np.newaxis])
        phase1.append(ph1[..., np.newaxis])

    # Save data using HDF5
    if save_as_h5:
        with h5py.File(h5_file, 'a') as h5f:
            for dataset_name, data in zip(['inputs', 'targets', 'phase0', 'phase1'], [i1, i2, ph0, ph1]):
                dset = h5f[dataset_name]
                dset.resize((dset.shape[0] + 1), axis=0)
                dset[-1] = data[..., np.newaxis]

    # Display images
    if show_images:
        i_range = [np.min(i1), np.max(i1)]
        ph_range = [np.min(ph0), np.max(ph0)]

        plt.figure(1, figsize=(10, 6))
        plt.clf()
        plt.suptitle(f'Image #{img_no}')

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

    print(f"{img_no}/{img_count}")

# Save data using NumPy
if save_as_npz:
    npz_file = f'{output_file}.npz'
    if save_disc_space:
        np.savez_compressed(npz_file, inputs=inputs, targets=targets, phase0=phase0, phase1=phase1)
    else:
        np.savez(npz_file, inputs=inputs, targets=targets, phase0=phase0, phase1=phase1)
    print(f"\nData has been saved as: {npz_file}")

# Close HDF5 file
if save_as_h5:
    h5f.close()
    print(f"\nData has been saved as: {h5_file}")

# Compress data file
if save_as_h5 and save_disc_space:
    with h5py.File(h5_file, 'r') as f_in:
        with h5py.File(f'{output_file}_compressed.h5', 'w') as f_out:
            for dataset_name in f_in:
                data = f_in[dataset_name][:]

                f_out.create_dataset(dataset_name, data=data, compression='gzip', compression_opts=9, dtype=data.dtype)

    print(f"File '{output_file}.h5' has been compressed.")

