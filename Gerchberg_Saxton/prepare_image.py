import numpy as np

def resize(image, nx, ny, px, py, x_pos, y_pos):
    image = image[x_pos:x_pos + 512, y_pos:y_pos + 512]

    # Convert to grayscale if needed
    if image.ndim == 3:
        image = np.mean(image, axis=-1)

    return image

