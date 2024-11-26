"""
The code implements the forward model for phase retrival from two defocused intensity images.
Your task is to solve the inverse problem,
that is to find the phase1 that will result in intensity2 prediction that is very close to the intensity2.

All distances are given in um
"""

import numpy as np
import os
import scipy.io as sio
from utils import propagate_as
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ####################################################################################################################
    # Simulation of the measurement data
    ####################################################################################################################
    # wavelength = 0.561  # wavelength of light
    wavelength = 0.1
    n0 = 1  # refractive index of air
    sampling = np.array([2.4, 2.4])     # sampling pitch of the camera
    z_vec = np.array([3.5e3, 11.5e3])   # locations of the measurement planes

    # Define the path to the npz file
    base_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska"
    filename = os.path.join(base_path, "training_data.npz")

    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # Load data from the npz file
    data_obj = np.load(filename)

    # Extract amplitude and phase data
    # Assuming `inputs` corresponds to amplitude and `phase0` corresponds to phase
    amp_obj = data_obj['inputs'][:, :, 0, 0]  # Load the first image as an example
    phase_obj = data_obj['phase0'][:, :, 0, 0]  # Load the first phase map as an example

    # Combine amplitude and phase into optical field
    u_obj = amp_obj * np.exp(1j*phase_obj)

    # Propagate to the first measurement plane
    u1 = propagate_as(u_obj,z_vec[0],wavelength,n0,sampling)
    # Propagate to the second measurement plane
    u2 = propagate_as(u_obj, z_vec[1], wavelength, n0, sampling)

    # Simulate the intensity images (i1 and i2 are the inputs for the phase retrival algorithm
    i1 = np.abs(u1) ** 2
    i2 = np.abs(u2) ** 2

    # Obtain the phase at z1 (phase1 is that we are looking for solving the inverse problem - tf.Variable)
    phase1 = np.angle(u1)

    ####################################################################################################################
    # The actual forward model
    ####################################################################################################################
    # we can start with null initial guess of the phase; gradient descent will iterative improve our guess
    phase1_est = np.zeros_like(phase1)
    # at the end we want phase1_est (estimate of phase1) to be as close to true phase1 as possible
    # phase1_est = phase1 # CHECK OUT THIS OPTION - UNCOMMENT THIS LINE

    # build optical field estimate
    u1_est = np.sqrt(i1) * np.exp(1j*phase1_est)

    # Propagate from the first to second measurement plane
    u2_est = propagate_as(u1_est,z_vec[1]-z_vec[0],wavelength,n0,sampling)

    # evaluate the intensity image corresponding to u2_est
    i2_est = np.abs(u2_est) ** 2

    # Display the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    im1 = ax1.imshow(i2_est)
    ax1.set_title("predicted intensity at z2")
    im2 = ax2.imshow(i2)
    ax2.set_title("the actual intensity at z2")
    plt.show()
