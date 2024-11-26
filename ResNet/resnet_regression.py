"""
ResNet-20 model
There are 3 groups. Each group has n=3 residual blocks. Each residual block has 2 Conv2D layers.
This relates to total number of 3*2*n + 2 = 20 layers.
The authors claim that the code worked only for SGD, not Adam or SGDW!
source:
https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-build-a-resnet-from-scratch-with-tensorflow-2-and-keras.md
how to get Tesnorboard type in terminal: python -m tensorboard.main --logdir=logs/
"""

import os
import numpy as np
import tensorflow
from tensorflow.image import flip_up_down
from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, \
    Conv2D, Lambda, Input, BatchNormalization, Activation, MaxPool2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def model_configuration():
    """
    Get configuration variables for the model.
    """

    # Load dataset for computing dataset size
    (input_train, input_labels, *rest) = load_dataset()

    # Generic config
    width, height, channels = 512, 512, 1
    batch_size = 1
    validation_split = 0.2  # 45/5 per the He et al. paper
    verbose = 1
    n = 3  # number of residual blocks in a single group
    init_fm_dim = 16  # initial number of feature maps; the number doubles when the feature map size halves.
    shortcut_type = "identity"  # or: projection

    # Dataset size
    train_size = (1 - validation_split) * len(input_train)
    val_size = validation_split * len(input_train)

    # Number of steps per epoch is dependent on batch size
    maximum_number_iterations = 32000  # per the He et al. paper
    # steps_per_epoch = np.ceil(train_size / batch_size).astype(int)
    steps_per_epoch = int(train_size / batch_size)
    # val_steps_per_epoch = np.ceil(val_size / batch_size).astype(int)
    val_steps_per_epoch = int(val_size / batch_size)
    # epochs = tensorflow.cast(tensorflow.math.ceil(maximum_number_iterations / steps_per_epoch), dtype=tensorflow.int64)
    epochs = int(maximum_number_iterations / steps_per_epoch)

    # Define loss function
    loss = tensorflow.keras.losses.MeanSquaredError()

    # Set layer init
    initializer = tensorflow.keras.initializers.HeNormal()

    # Define optimizer
    lr_schedule = ExponentialDecay(
        initial_learning_rate=5e-2,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)

    # Load Tensorboard callback
    tensorboard = TensorBoard(
        os.path.join(os.getcwd(), "logs"),
        histogram_freq=1,
        write_steps_per_second=True,
        write_images=True,
        update_freq='epoch'
    )

    # Save a model checkpoint after every epoch
    checkpoint = ModelCheckpoint(
        os.path.join(os.getcwd(), "model_checkpoint.keras"),
        save_freq="epoch"
    )

    # Add callbacks to list
    callbacks = [
        tensorboard,
        checkpoint
    ]

    # Create config dictionary
    config = {
        "epochs": epochs, # new word added
        "width": width,
        "height": height,
        "dim": channels,
        "batch_size": batch_size,
        "validation_split": validation_split,
        "verbose": verbose,
        "stack_n": n,
        "initial_num_feature_maps": init_fm_dim,
        "training_ds_size": train_size,
        "steps_per_epoch": steps_per_epoch,
        "val_steps_per_epoch": val_steps_per_epoch,
        "num_epochs": epochs,
        "loss": loss,
        "optim": optimizer,
        "initializer": initializer,
        "callbacks": callbacks,
        "shortcut_type": shortcut_type
    }

    return config


def load_dataset():
    """
    Load the dataset from an .npz file containing inputs and targets.

    Parameters:
    - file_path (str): Path to the .npz file.

    Returns:
    - tuple: A tuple containing:
        - inputs (numpy array): Array of input images or data.
        - targets (numpy array): Array of target images or data.
    """
    try:
        # Load the .npz file
        data = np.load("C:/Users/Monika Walocha/Desktop/adek files/_python/praca_inzynierska/training_data_NOWE_40.npz")

        # Extract the inputs and targets
        inputs = data['inputs']
        targets = data['targets']

        # Ensure data consistency (optional)
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Number of inputs and targets do not match.")

        print(f"Loaded dataset: {inputs.shape[0]} samples.")
        return inputs, targets

    except FileNotFoundError:
        raise FileNotFoundError(f"The file was not found.")
    except KeyError as e:
        raise KeyError(f"Missing expected data key in the file: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the dataset: {e}")


# def preprocessed_dataset():
#     """
#     Load and preprocess the CIFAR-10 dataset.
#     """
#     inputs, targets = load_dataset()
#
#     # (input_train, target_train), (input_test, target_test) = load_dataset()
#
#     train_dataset = tensorflow.data.Dataset.from_tensor_slices((inputs, targets))
#     train_dataset = train_dataset.shuffle(buffer_size=inputs.shape[0]).batch(1).prefetch(
#         buffer_size=tensorflow.data.experimental.AUTOTUNE)
#
#     # input_train = input_train[:, :, :, 0]
#     # input_test = input_test[:, :, :, 0]
#     # input_train = input_train[:, :, :, np.newaxis]
#     # input_test = input_test[:, :, :, np.newaxis]
#
#     config = model_configuration()
#
#     # Data generator for training data
#     train_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(
#         validation_split=config.get("validation_split"),
#         horizontal_flip=True,
#         rescale=1. / 255
#         # preprocessing_function=tensorflow.keras.applications.resnet50.preprocess_input
#     )
#     target_train = flip_up_down(input_train)
#     target_test = flip_up_down(input_test)
#
#     # Generate training and validation batches
#     train_batches = train_generator.flow(input_train, target_train, batch_size=config.get("batch_size"),
#                                          subset="training")
#     validation_batches = train_generator.flow(input_train, target_train, batch_size=config.get("batch_size"),
#                                               subset="validation")
#
#     # Data generator for testing data
#     test_generator = tensorflow.keras.preprocessing.image.ImageDataGenerator(
#         # preprocessing_function=tensorflow.keras.applications.resnet50.preprocess_input,
#         rescale=1. / 255
#         )
#
#     # Generate test batches
#     test_batches = test_generator.flow(input_test, target_test, batch_size=config.get("batch_size"))
#
#     return train_batches, validation_batches, test_batches


def preprocessed_dataset():
    """
    Preprocess and split the dataset into training, validation, and test sets.

    Returns:
    - train_dataset: Dataset for training.
    - validation_dataset: Dataset for validation.
    - test_dataset: Dataset for testing.
    """
    # Load the dataset
    inputs, targets = load_dataset()

    # Get configuration
    config = model_configuration()
    validation_split = config["validation_split"]
    batch_size = config["batch_size"]

    # Calculate split indices
    total_samples = len(inputs)
    val_size = int(total_samples * validation_split)
    train_size = total_samples - val_size

    # Split dataset
    train_inputs, val_inputs = inputs[:train_size], inputs[train_size:]
    train_targets, val_targets = targets[:train_size], targets[train_size:]

    # Create TensorFlow datasets
    train_dataset = tensorflow.data.Dataset.from_tensor_slices((train_inputs, train_targets))
    validation_dataset = tensorflow.data.Dataset.from_tensor_slices((val_inputs, val_targets))

    # Shuffle, batch, and prefetch datasets
    train_dataset = train_dataset.shuffle(train_size).batch(batch_size).prefetch(buffer_size=tensorflow.data.AUTOTUNE)
    validation_dataset = validation_dataset.batch(batch_size).prefetch(buffer_size=tensorflow.data.AUTOTUNE)

    # Use the last part of the training set as the test set
    test_dataset = validation_dataset.take(val_size // 2)
    validation_dataset = validation_dataset.skip(val_size // 2)

    return train_dataset, validation_dataset, test_dataset


def residual_block(x, number_of_filters):
    """
    Residual block with
    """
    # Retrieve initializer
    config = model_configuration()
    initializer = config.get("initializer")

    # Create skip connection
    x_skip = x

    # Perform the original mapping
    x = Conv2D(number_of_filters, kernel_size=(3, 3), strides=(1, 1),
               kernel_initializer=initializer, padding="same")(x_skip)
    x = BatchNormalization(axis=3)(x)
    x = Activation("relu")(x)
    x = Conv2D(number_of_filters, kernel_size=(3, 3),
               kernel_initializer=initializer, padding="same")(x)
    x = BatchNormalization(axis=3)(x)

    # Add the skip connection to the regular mapping
    x = Add()([x, x_skip])

    # Nonlinearly activate the result
    x = Activation("relu")(x)

    # Return the result
    return x


def ResidualBlocks(x):
    """
    Set up the residual blocks.
    """
    # Retrieve values
    config = model_configuration()

    # Set initial filter size
    filter_size = config.get("initial_num_feature_maps")

    # Paper: "Then we use a stack of 6n layers (...)
    #	with 2n layers for each feature map size."
    # 6n/2n = 3, so there are always 3 groups.
    for layer_group in range(4):
        # Each block in our code has 2 weighted layers,
        # and each group has 2n such blocks,
        # so 2n/2 = n blocks per group.
        for block in range(config.get("stack_n")):
            x = residual_block(x, filter_size)
    # Return final layer
    return x


def ResNetPath(x, scale=2):
    """
    Define a single ResNet path with rescaled feature maps size
    Args:
        x: input feature map
        scale: factor at which the feature maps should be rescaled before entering to Residual Block

    Returns: output feature map with the original size

    """
    config = model_configuration()
    initializer = model_configuration().get("initializer")

    assert (scale >= 1 and isinstance(scale,int))
    if scale > 1:
        x = MaxPool2D(pool_size=(3, 3), strides=scale, padding="same")(x)
    x = ResidualBlocks(x)
    if scale > 1:
        x = UpSampling2D(size=scale, interpolation="bilinear")(x)
    x = Conv2D(config.get("initial_num_feature_maps"), kernel_size=(3, 3),
               strides=(1, 1), kernel_initializer=initializer, padding="same")(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def model_base(shp):
    """
    Base structure of the model, with residual blocks
    attached.
    """
    # Get number of classes from model configuration
    config = model_configuration()
    initializer = model_configuration().get("initializer")

    # Define model structure
    inputs = Input(shape=shp)
    x = Conv2D(config.get("initial_num_feature_maps"), kernel_size=(3, 3),
               strides=(1, 1), kernel_initializer=initializer, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x1 = ResNetPath(x, 1)
    x2 = ResNetPath(x, 2)
    x = Add()([x1, x2])
    x = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1),
               kernel_initializer=initializer, padding="same")(x)
    x = BatchNormalization()(x)
    outputs = Activation("relu")(x)

    return inputs, outputs


def init_model():
    """
    Initialize a compiled ResNet model.
    """
    # Get shape from model configuration
    config = model_configuration()

    # Get model base
    inputs, outputs = model_base((config.get("width"), config.get("height"),
                                  config.get("dim")))

    # Initialize and compile model
    model = Model(inputs, outputs, name=config.get("name"))

    model.compile(loss=config.get("loss"),
                  optimizer=config.get("optim"),
                  metrics=config.get("optim_additional_metrics"))

    # Print model summary
    model.summary()

    return model


def train_model(model, train_batches, validation_batches):
    """
    Train an initialized model.
    """

    # Get model configuration
    config = model_configuration()
    ##########################################################################################
    # train_batches = train_batches.shuffle(buffer_size=1000).repeat()
    # train_batches = train_batches.repeat(config.get("epochs"))
    # train_batches = train_batches.repeat()
    ##########################################################################################

    # Fit data to model
    hist_obj = model.fit(train_batches,
                  batch_size=config.get("batch_size"),
                  epochs=config.get("num_epochs"),
                  verbose=config.get("verbose"),
                  callbacks=config.get("callbacks"),
                  # steps_per_epoch=config.get("steps_per_epoch"),
                  validation_data=validation_batches,
                  # validation_steps=config.get("val_steps_per_epoch"))
                         )

    loss = hist_obj.history['loss']
    val_loss = hist_obj.history['val_loss']
    plt.plot(np.log(loss), label='Training Loss')
    plt.plot(np.log(val_loss), label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Log loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    return model


def evaluate_model(model, test_batches):
    """
    Evaluate a trained model.
    """
    # Evaluate model
    score = model.evaluate(test_batches, verbose=1)
    print(f'Test loss: {score}')


def training_process():
    """
    Run the training process for the ResNet model.
    """

    # Get dataset
    train_batches, validation_batches, test_batches = preprocessed_dataset()

    # Preport directory for logs
    log_dir = os.path.join(os.getcwd(), "logs")
    if os.path.exists(log_dir) and os.path.isdir(log_dir):
         shutil.rmtree(log_dir)

    # Initialize ResNet
    resnet = init_model()

    # Train ResNet model
    trained_resnet = train_model(resnet, train_batches, validation_batches)

    # Evaluate trained ResNet model post training
    evaluate_model(trained_resnet, test_batches)

if __name__ == "__main__":
    training_process()