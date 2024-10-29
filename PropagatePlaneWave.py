import numpy as np


def propagate_plane_wave(uin, z, n0, lambda_, dx, dy):
    """
        Propagate an optical field along the optical axis in a homogenous medium
        using the angular spectrum method.

        Wykorzystanie metody spektrum kątowego do symulacji propagacji fali optycznej w jednorodnym ośrodku

        IN:
        uin - pole wejściowe dla 'z'=0
        z - odległość propagacji
        n0 - współczynnik załamania ośrodka
        lambda - długość fali
        dx, dy - interwały próbkowania

        OUT:
        uout - pole wyjściowe w odległości 'z'
        """
    Ny, Nx = uin.shape # przechowuje wymiary pola wejściowego w pixelach
    k = 2 * np.pi / lambda_ # liczba falowa

    dfx = 1 / (Nx * dx) # rozdzielczość częstotliwości w kierunku 'x'
    fx = np.fft.fftfreq(Nx, d=dx) # częstotliwość próbkowania w kierunku 'x'

    dfy = 1 / (Ny * dy) # rozdzielczość częstotliwości w kierunku 'y'
    fy = np.fft.fftfreq(Ny, d=dy) # częstotliwość próbkowania w kierunku 'y'

    Fx, Fy = np.meshgrid(fx, fy) # siatka częstości pozwala na operacje w domenie częstotliwości


    if z < 0:
        uin = np.conj(uin)


    FTu = np.fft.fft2(uin) # dwuwymiarowa transformacja fouriera na polu wejściowym

    Kernel = np.exp(1j * k * abs(z) * np.real(np.sqrt(n0 ** 2 - lambda_ ** 2 * (Fx ** 2 + Fy ** 2))))
    Kernel[n0 ** 2 < lambda_ ** 2 * (Fx ** 2 + Fy ** 2)] = 0

    FTu *= Kernel

    uout = np.fft.ifft2(FTu) # odrotna dwuwymiarowa transformacja fouriera

    if z < 0:
        uout = np.conj(uout)

    return uout


'''
uout - macierz liczb zespolonych, liczby reprezentują amplitudę i fazę w różnych punktach przestrzeni dla 'z'
wymiary uout będą takie same jak wymiary pola wejściowego (Nx, Ny)
wartości przedstawiają jak pole optyczne zmienia się po propagacji na odległość 'z'
w czasie propagacji mogą zachodzić zjawiska takie jak: rozpraszanie, interferencja, zmiana kształtu
'''

