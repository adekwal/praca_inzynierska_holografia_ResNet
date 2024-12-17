from resnet_regression import init_model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# initialize model
# model = init_model()
# # load model
# model.load_weights('trained_net.weights.h5')
model = tf.keras.models.load_model('1000images_1epoch_model_checkpoint.keras')

try:
    # Load the .npz file
    data = np.load(
        "C:/Users/jw/Desktop/dyplomy/Adam Walocha/AdamWalochaGitRep/praca_inzynierska_ResNet/training_data_100.npz")

    # Extract the inputs and targets
    i1 = data['inputs'][10, :, :, 0]
    i2 = data['targets'][10, :, :, 0]

except FileNotFoundError:
    raise FileNotFoundError("The dataset file was not found.")
except KeyError as e:
    raise KeyError(f"Missing expected data key in the file: {e}")
except Exception as e:
    raise RuntimeError(f"An error occurred while loading the dataset: {e}")

input_data = i1[np.newaxis, :, :, np.newaxis]
i2_predicted = model.predict(input_data, verbose=3) # Expected shape (None, 512, 512, 1)
i2_predicted = np.squeeze(i2_predicted)

fig1, axs = plt.subplots(1, 3, figsize=(15, 8))
im1 = axs[0].imshow(i1)
fig1.colorbar(im1, ax=axs[0])
axs[0].set_title("Intensity at z1")

im2 = axs[1].imshow(i2)
fig1.colorbar(im2, ax=axs[1])
axs[1].set_title("Intensity at z2")

im2 = axs[2].imshow(i2_predicted)
fig1.colorbar(im2, ax=axs[2])
axs[2].set_title("Predicted intensity at z2")
plt.show()
