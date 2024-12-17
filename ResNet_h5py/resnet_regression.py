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
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, \
    Conv2D, Lambda, Input, BatchNormalization, Activation, MaxPool2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import shutil
import h5py

dataset_file = r"C:/Users/Monika Walocha/Desktop/adek files/_python/praca_inzynierska/data250.h5"
global_dataset = None

def get_dataset_size(file_path):
    with h5py.File(file_path, 'r') as h5f:
        dataset = h5f['inputs']
        return dataset.shape[0]

def model_configuration():
    """
    Get configuration variables for the model.
    """
    global global_dataset  # Use the global dataset variable

    # Ensure the dataset is loaded
    if global_dataset is None:
        load_dataset()

    # Generic configuration
    width, height, channels = 512, 512, 1
    batch_size = 1
    validation_split = 0.2
    verbose = 1
    n = 3  # Number of residual blocks in a single group
    init_fm_dim = 16  # Initial number of feature maps; doubles as the feature map size halves
    shortcut_type = "identity"  # Shortcut type: "identity" or "projection"

    ## WARZONE DO NOT CROSS THE BORDER ##

    num_samples = get_dataset_size(dataset_file)
    # print(f"H5: {num_samples}")
    #
    # data_numpy = np.load("C:/Users/Monika Walocha/Desktop/adek files/_python/praca_inzynierska/data100.npz")
    # inputs_numpy = data_numpy['inputs']
    # print(f"NPZ: {inputs_numpy.shape[0]}")

    train_size = (1 - validation_split) * num_samples
    val_size = validation_split * num_samples

    ## WARZONE DO NOT CROSS THE BORDER ##

    # Calculate steps per epoch based on dataset size and batch size
    maximum_number_iterations = 800  # Maximum number of iterations as per the paper (32000)
    steps_per_epoch = np.ceil(train_size / batch_size).astype(int)
    val_steps_per_epoch = np.ceil(val_size / batch_size).astype(int)
    epochs = tensorflow.cast(
        tensorflow.math.ceil(maximum_number_iterations / steps_per_epoch),
        dtype=tensorflow.int64
    )

    # Define the loss function
    loss = tensorflow.keras.losses.MeanSquaredError()

    # Set layer initializer
    initializer = tensorflow.keras.initializers.HeNormal()

    # Define the optimizer
    lr_schedule = ExponentialDecay(
        initial_learning_rate=5e-2,
        decay_steps=10000,
        decay_rate=0.9
    )
    optimizer = Adam(learning_rate=lr_schedule)

    # TensorBoard callback for monitoring
    tensorboard = TensorBoard(
        os.path.join(os.getcwd(), "logs"),
        histogram_freq=1,
        write_steps_per_second=True,
        write_images=True,
        update_freq='epoch'
    )

    # Model checkpoint callback for saving weights
    checkpoint = ModelCheckpoint(
        os.path.join(os.getcwd(), "model_checkpoint.keras"),
        save_freq="epoch"
    )

    # Add callbacks to a list
    callbacks = [
        tensorboard,
        checkpoint
    ]

    # Create configuration dictionary
    config = {
        "epochs": epochs,
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
    global global_dataset

    try:
        h5f = h5py.File(dataset_file, "r")
        if 'inputs' not in h5f or 'targets' not in h5f:
            raise KeyError("Missing 'inputs' or 'targets' datasets in the HDF5 file.")

        # Pointers
        global_dataset = {"inputs": h5f['inputs'], "targets": h5f['targets']}

        # Number of samples
        num_samples = global_dataset['inputs'].shape[0]
        print(f"Loaded dataset: {num_samples} samples into global_dataset.")

    except FileNotFoundError:
        raise FileNotFoundError("The dataset file was not found.")
    except KeyError as e:
        raise KeyError(f"Missing expected data key in the file: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the dataset: {e}")

## WARZONE DO NOT CROSS THE BORDER ##

def data_generator(inputs, targets, batch_size):
    """
    Generator function for HDF5-based data. Yields batches of data.
    """
    total_samples = inputs.shape[0]
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_inputs = inputs[start_idx:end_idx]
        batch_targets = targets[start_idx:end_idx]
        yield batch_inputs, batch_targets


def preprocessed_dataset():
    global global_dataset

    if global_dataset is None:
        load_dataset()

    inputs = global_dataset["inputs"]
    targets = global_dataset["targets"]

    config = model_configuration()
    validation_split = config["validation_split"]  # np. 0.2 (20% na validation + test)
    batch_size = config["batch_size"]

    total_samples = len(inputs)
    val_test_size = int(total_samples * validation_split)
    val_size = val_test_size // 2
    train_size = total_samples - val_test_size

    # Calculate index
    train_indices = range(0, train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_samples)

    # Define generators
    train_gen = lambda: data_generator(inputs[train_indices], targets[train_indices], batch_size)
    val_gen = lambda: data_generator(inputs[val_indices], targets[val_indices], batch_size)
    test_gen = lambda: data_generator(inputs[test_indices], targets[test_indices], batch_size)

    # Create dataset
    train_dataset = tensorflow.data.Dataset.from_generator(
        train_gen,
        output_signature=(
            tensorflow.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=inputs.dtype),
            tensorflow.TensorSpec(shape=(None,) + targets.shape[1:], dtype=targets.dtype),
        )
    )

    validation_dataset = tensorflow.data.Dataset.from_generator(
        val_gen,
        output_signature=(
            tensorflow.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=inputs.dtype),
            tensorflow.TensorSpec(shape=(None,) + targets.shape[1:], dtype=targets.dtype),
        )
    )

    test_dataset = tensorflow.data.Dataset.from_generator(
        test_gen,
        output_signature=(
            tensorflow.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=inputs.dtype),
            tensorflow.TensorSpec(shape=(None,) + targets.shape[1:], dtype=targets.dtype),
        )
    )

    train_dataset = train_dataset.shuffle(buffer_size=1024).repeat().prefetch(tensorflow.data.AUTOTUNE)
    validation_dataset = validation_dataset.repeat().prefetch(tensorflow.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tensorflow.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset

# def preprocessed_dataset():
#     """
#     Preprocess the global dataset into TensorFlow datasets for training, validation, and testing.
#     Uses global `global_dataset`.
#     """
#     global global_dataset  # Use the global dataset variable
#
#     # Ensure the dataset is loaded
#     if global_dataset is None:
#         load_dataset()
#
#     # Access inputs and targets from the global dataset
#     inputs = global_dataset["inputs"]
#     targets = global_dataset["targets"]
#
#     # Get configuration
#     config = model_configuration()
#     validation_split = config["validation_split"]
#     batch_size = config["batch_size"]
#
#     # Ensure batch_size is valid
#     if batch_size <= 0:
#         raise ValueError("batch_size must be greater than zero.")
#
#     # Calculate dataset split sizes
#     total_samples = len(inputs)
#     val_size = int(total_samples * validation_split)
#     train_size = total_samples - val_size
#
#     # Split dataset into indices for training and validation
#     train_indices = range(train_size)
#     val_indices = range(train_size, total_samples)
#
#     # Ensure test batch size is valid
#     test_batch_size = max(batch_size // 2, 1)
#
#     # Log key details for debugging
#     print(f"Dataset summary:")
#     print(f"  Total samples: {total_samples}")
#     print(f"  Training samples: {train_size}")
#     print(f"  Validation samples: {val_size}")
#     print(f"  Batch size: {batch_size}, Test batch size: {test_batch_size}")
#
#     # Define data generators
#     train_gen = lambda: data_generator(inputs[train_indices], targets[train_indices], batch_size)
#     val_gen = lambda: data_generator(inputs[val_indices], targets[val_indices], batch_size)
#     test_gen = lambda: data_generator(inputs[val_indices], targets[val_indices], test_batch_size)
#
#     # Create TensorFlow Dataset objects from generators
#     train_dataset = tensorflow.data.Dataset.from_generator(
#         train_gen,
#         output_signature=(
#             tensorflow.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=inputs.dtype),
#             tensorflow.TensorSpec(shape=(None,) + targets.shape[1:], dtype=targets.dtype),
#         ),
#     )
#
#     validation_dataset = tensorflow.data.Dataset.from_generator(
#         val_gen,
#         output_signature=(
#             tensorflow.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=inputs.dtype),
#             tensorflow.TensorSpec(shape=(None,) + targets.shape[1:], dtype=targets.dtype),
#         ),
#     )
#
#     test_dataset = tensorflow.data.Dataset.from_generator(
#         test_gen,
#         output_signature=(
#             tensorflow.TensorSpec(shape=(None,) + inputs.shape[1:], dtype=inputs.dtype),
#             tensorflow.TensorSpec(shape=(None,) + targets.shape[1:], dtype=targets.dtype),
#         ),
#     )
#
#     # Shuffle, repeat, and prefetch datasets for efficiency
#     train_dataset = train_dataset.shuffle(buffer_size=1024).repeat().prefetch(tensorflow.data.AUTOTUNE)
#     validation_dataset = validation_dataset.repeat().prefetch(tensorflow.data.AUTOTUNE)
#     test_dataset = test_dataset.repeat().prefetch(tensorflow.data.AUTOTUNE)
#
#     # Return processed datasets
#     return train_dataset, validation_dataset, test_dataset



## WARZONE DO NOT CROSS THE BORDER ##



# def preprocessed_dataset():
#     """
#     Preprocess and split the dataset into training, validation, and test sets.
#
#     Returns:
#     - train_dataset: Dataset for training.
#     - validation_dataset: Dataset for validation.
#     - test_dataset: Dataset for testing.
#     """
#     global global_dataset  # Use the global dataset variable
#
#     # Ensure the dataset is loaded
#     if global_dataset is None:
#         load_dataset()
#
#     # Access inputs and targets from the global dataset
#     inputs = global_dataset["inputs"]
#     targets = global_dataset["targets"]
#
#     # Get configuration
#     config = model_configuration()
#     validation_split = config["validation_split"]
#     batch_size = config["batch_size"]
#
#     # Calculate split indices
#     total_samples = len(inputs)
#     val_size = int(total_samples * validation_split)
#     train_size = total_samples - val_size
#
#     # Split dataset
#     train_inputs, val_inputs = inputs[:train_size], inputs[train_size:]
#     train_targets, val_targets = targets[:train_size], targets[train_size:]
#
#     # Create TensorFlow datasets
#     train_dataset = tensorflow.data.Dataset.from_tensor_slices((train_inputs, train_targets))
#     validation_dataset = tensorflow.data.Dataset.from_tensor_slices((val_inputs, val_targets))
#
#     # Shuffle, batch, and prefetch datasets
#     train_dataset = train_dataset.shuffle(train_size).batch(batch_size).prefetch(buffer_size=tensorflow.data.AUTOTUNE)
#     validation_dataset = validation_dataset.batch(batch_size).prefetch(buffer_size=tensorflow.data.AUTOTUNE)
#
#     # Use the last part of the training set as the test set
#     test_dataset = validation_dataset.take(val_size // 2)
#     validation_dataset = validation_dataset.skip(val_size // 2)
#
#     return train_dataset, validation_dataset, test_dataset


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
                  steps_per_epoch=config.get("steps_per_epoch"), #
                  validation_data=validation_batches,
                  validation_steps=config.get("val_steps_per_epoch") #
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