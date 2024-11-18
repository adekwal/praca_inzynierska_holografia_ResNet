import numpy as np

def propagate_plane_wave(uin, z, n0, lambda_, dx, dy):

    Ny, Nx = uin.shape
    k = 2 * np.pi / lambda_

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

