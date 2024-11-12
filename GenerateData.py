import numpy as np
from PropagatePlaneWave import propagate_plane_wave
from scipy.ndimage import gaussian_filter, zoom

def generate_data(ph0, delta_ph_max, z1, z2, lambda_, dx, nx, ny, px, py):
    sigma = 85

    ## Generation of object wave
    delta_ph = delta_ph_max
    #delta_ph = (2*np.random.rand()-1)*delta_ph_max
    ph0 = ph0[10:-10, 10:-10]
    #ph0 = ph0[19:-19, 16:-16]

    # skalowanie i przetwarzanie obrazu
    zoom_factor_x = nx/ph0.shape[1]
    zoom_factor_y = ny/ph0.shape[0]

    ## GRAY
    if ph0.ndim == 3:
        ph0 = np.mean(ph0, axis=-1)

    ph0 = zoom(ph0, (zoom_factor_y, zoom_factor_x), order=1)

    '''
    zoom_factor_z = 1
    print("Shape of ph0:", ph0.shape)
    ## RGB
    ph0 = zoom(ph0, (zoom_factor_y, zoom_factor_x, zoom_factor_z), order=1)
    '''

    # filtracja górnoprzepustowa
    ph0 = ph0 - gaussian_filter(ph0, sigma)

    # normalizacja obrazu
    '''
    ph0 = np.mean(ph0, axis = -1) if ph0.ndim == 3 else ph0 # jeśli obraz jest RGB (a nie jest)
    '''
    ph0 = (ph0 - np.min(ph0)) / (np.max(ph0) - np.min(ph0))  # normalizacja fazy do zakresu (0,1)

    # zastosowanie zmian fazy
    ph0 = ph0 * delta_ph

    # dodawanie paddingu
    pad_x = (int(py/2-ny/2), int(py/2-ny/2))
    pad_y = (int(px/2-nx/2), int(px/2-nx/2))
    ph0 = np.pad(ph0, (pad_x, pad_y), mode='edge')

    # obliczenie pola obiektu
    ph0 = ph0 - np.median(ph0)
    u_obj = np.exp(1j * ph0)


    ## Simulation of two defocused intensity measurements
    # Symulacja pomiarów intensywności w dwóch odległościach
    u1 = propagate_plane_wave(u_obj, z1, 1, lambda_, dx, dx)
    u2 = propagate_plane_wave(u_obj, z2, 1, lambda_, dx, dx)

    # Obliczenie intensywności i fazy
    i1 = np.abs(u1)**2 # intensywność to kwadrat modułu pola
    i2 = np.abs(u2)**2
    ph1 = np.angle(u1) # faza to kąt liczby zespolonej
    ph2 = np.angle(u2)

    return i1, i2, ph1, ph2, ph0

