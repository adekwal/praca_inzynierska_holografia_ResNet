import numpy as np

def propagate_plane_wave(uin, z, n0, lambda_, dx, dy):

    Ny, Nx = uin.shape
    k = 2 * np.pi / lambda_

    # Dokładnie odwzorowana siatka częstotliwości zgodna z MATLAB-em
    dfx = 1 / (Nx * dx)
    fx = np.concatenate((np.arange(0, Nx // 2), np.arange(-Nx // 2, 0))) * dfx
    dfy = 1 / (Ny * dy)
    fy = np.concatenate((np.arange(0, Ny // 2), np.arange(-Ny // 2, 0))) * dfy

    Fx, Fy = np.meshgrid(fx, fy)

    if z < 0:
        uin = np.conj(uin)

    FTu = np.fft.fft2(uin)

    Kernel = np.exp(1j * k * abs(z) * np.real(np.sqrt(n0**2 - lambda_**2 * (Fx**2 + Fy**2))))
    Kernel[n0**2 < lambda_**2 * (Fx**2 + Fy**2)] = 0

    FTu *= Kernel

    uout = np.fft.ifft2(FTu)

    if z < 0:
        uout = np.conj(uout)

    return uout

'''
import math
import numpy as np


def propagate_plane_wave(ui, z, n0, lambda_, dx, dy):

    k = 2 * math.pi / lambda_
    Ny, Nx = ui.shape
    dfx = 1 / (Nx * dx)
    dfy = 1 / (Ny * dy)
    fx = np.fft.fftshift(np.arange(Nx) -Nx / 2) * dfx
    fy = np.fft.fftshift(np.arange(Ny) - Ny / 2) * dfy
    fx2 = fx ** 2
    fy2 = fy ** 2

    fy2_2d = np.dot(np.ones([Nx, 1]), fy2[np.newaxis])
    fx2_2d = np.dot(fx2[:, np.newaxis], np.ones([1, Ny]))
    under_sqrt = np.power(n0, 2) - np.power(lambda_, 2) * (fx2_2d + fy2_2d)

    under_sqrt[under_sqrt < 0] = 0
    phase_kernel = k * np.abs(z) * np.sqrt(under_sqrt)

    if z < 0:
        ui = np.conj(ui)

    ftu = np.fft.fft2(ui) * np.exp(1j * phase_kernel)
    uo = np.fft.ifft2(ftu)

    if z < 0:
        uo = np.conj(uo)

    return uo
'''
