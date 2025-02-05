import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

model = tf.keras.models.load_model(
    r"C:\Users\Monika Walocha\Desktop\adek files\_python\ResNet\trained_models\epoch_10_model_checkpoint.keras" # set the path to trained ResNet model
)

try:
    with h5py.File(
        r"C:/Users/Monika Walocha/Desktop/adek files/_python/praca_inzynierska/dane50_compressed.h5", # set the path to training dataset file
        "r",
    ) as file:
        inputs = file["inputs"][:]
        targets = file["targets"][:]

    i1 = inputs[8, :, :, 0]
    i2 = targets[8, :, :, 0]

    input_data = i1[np.newaxis, :, :, np.newaxis]

    i2_predicted = model.predict(input_data, verbose=3) # Expected shape (None, 512, 512, 1)
    i2_predicted = np.squeeze(i2_predicted)

    vmin = min(i1.min(), i2.min(), i2_predicted.min())
    vmax = max(i1.max(), i2.max(), i2_predicted.max())

    fig1, axs = plt.subplots(1, 3, figsize=(15, 8))
    im1 = axs[0].imshow(i1, vmin=vmin, vmax=vmax)
    axs[0].set_title("Intensity at z1")

    im2 = axs[1].imshow(i2, vmin=vmin, vmax=vmax)
    axs[1].set_title("Intensity at z2")

    im3 = axs[2].imshow(i2_predicted, vmin=vmin, vmax=vmax)
    axs[2].set_title("Predicted intensity at z2")

    cbar = fig1.colorbar(im3, ax=axs, location='right', shrink=0.8)
    cbar.set_label("Intensity Value")

    plt.show()

except FileNotFoundError:
    raise FileNotFoundError("The dataset file was not found.")
except KeyError as e:
    raise KeyError(f"Missing expected data key in the file: {e}")
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the dataset: {e}")

