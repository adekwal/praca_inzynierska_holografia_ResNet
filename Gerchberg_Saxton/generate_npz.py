import h5py
import numpy as np

h5_path = r"C:\Users\Monika Walocha\Desktop\adek files\_python\praca_inzynierska\dane50_compressed.h5"
filename = "obraz"
index = 0

# Load HDF5 file
with h5py.File(h5_path, 'r') as h5_file:
    inputs = h5_file['inputs']
    i1 = inputs[index]

# Save as npz file
np.savez(filename + ".npz", img=i1)
print(f"Image has been saved as '{filename}.npz'")
