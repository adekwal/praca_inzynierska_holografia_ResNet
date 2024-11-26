import math
import numpy as np


def propagate_as(ui, z, wavelength, n0, sampling):
    """Propagate optical field to another parallel plane using angular spectrum method (as)

    In the applied convention the optical field is given in x-y plane, the source and target plane are displaced by z

    Parameters:
    __________
    ui : 2D complex ndarray
        optical field to be propagated
    z : float
        propagation distance
    wavelength :float
        wavelength of light in a vacuum
    n0 : float
        refractive index of a surrounding medium
    sampling : ndarray [sampling in x, sampling in y]
        sampling interval for x and y directions

    Returns:
    ________
    uo : 2D complex ndarray
        optical field after propagation
    """

    k = 2 * math.pi / wavelength
    size_pxl = ui.shape
    dfx = 1 / size_pxl[0] / sampling[0]
    dfy = 1 / size_pxl[1] / sampling[1]
    fx = np.fft.fftshift(np.arange(size_pxl[0]) - size_pxl[0] / 2) * dfx
    fy = np.fft.fftshift(np.arange(size_pxl[1]) - size_pxl[1] / 2) * dfy
    fx2 = fx ** 2
    fy2 = fy ** 2

    fy2_2d = np.dot(np.ones([size_pxl[0], 1]), fy2[np.newaxis])
    fx2_2d = np.dot(fx2[:, np.newaxis], np.ones([1, size_pxl[1]]))
    under_sqrt = np.power(n0, 2) - np.power(wavelength, 2) * (fx2_2d + fy2_2d)

    under_sqrt[under_sqrt < 0] = 0
    phase_kernel = k * np.abs(z) * np.sqrt(under_sqrt)

    if z < 0:
        ui = np.conj(ui)

    ftu = np.fft.fft2(ui) * np.exp(1j * phase_kernel)
    uo = np.fft.ifft2(ftu)

    if z < 0:
        uo = np.conj(uo)

    return uo


# def image_with_colorbar(img, ax, fig, title, **params):
#     im = ax.imshow(img, **params)
#     ax.set_title(title)
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes('right', size='5%', pad=0.05)
#     fig.colorbar(im, cax, orientation='vertical')