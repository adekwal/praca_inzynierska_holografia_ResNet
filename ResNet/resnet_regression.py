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
from tensorflow.image import flip_up_down
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, \
    Conv2D, Lambda, Input, BatchNormalization, Activation, MaxPool2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Global variable to store the dataset
global_dataset = None

def model_configuration():
    """
    Get configuration variables for the model.
    """
    global global_dataset  # Use the global dataset variable

    # Ensure the dataset is loaded
    if global_dataset is None:
        load_dataset()

    # Access inputs from the global dataset
    inputs = global_dataset["inputs"]

    # Generic configuration
    width, height, channels = 512, 512, 1
    batch_size = 1
    validation_split = 0.2  # Validation split ratio (e.g., 20%)
    verbose = 1
    n = 3  # Number of residual blocks in a single group
    init_fm_dim = 16  # Initial number of feature maps; doubles as the feature map size halves
    shortcut_type = "identity"  # Shortcut type: "identity" or "projection"

    # Dataset size
    train_size = (1 - validation_split) * len(inputs)  # Training dataset size
    val_size = validation_split * len(inputs)  # Validation dataset size

    # Calculate steps per epoch based on dataset size and batch size
    maximum_number_iterations = 320  # Maximum number of iterations as per the paper
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
    """
    Load the dataset from an .npz file into a global variable.

    The function assigns the inputs and targets to the global variable `global_dataset`.

    Raises:
        - FileNotFoundError: If the dataset file is not found.
        - KeyError: If required keys are missing in the .npz file.
        - RuntimeError: If another error occurs during loading.
    """
    global global_dataset  # Declare the global variable

    try:
        # Load the .npz file
        data = np.load("C:/Users/Monika Walocha/Desktop/adek files/_python/praca_inzynierska/training_data_NOWE_40.npz")

        # Extract the inputs and targets
        inputs = data['inputs']
        targets = data['targets']

        # Ensure data consistency
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("Number of inputs and targets do not match.")

        # Assign to the global variable
        global_dataset = {"inputs": inputs, "targets": targets}

        print(f"Loaded dataset: {inputs.shape[0]} samples into global_dataset.")

    except FileNotFoundError:
        raise FileNotFoundError("The dataset file was not found.")
    except KeyError as e:
        raise KeyError(f"Missing expected data key in the file: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the dataset: {e}")


def preprocessed_dataset():
    """
    Preprocess and split the dataset into training, validation, and test sets.

    Returns:
    - train_dataset: Dataset for training.
    - validation_dataset: Dataset for validation.
    - test_dataset: Dataset for testing.
    """
    global global_dataset  # Use the global dataset variable

    # Ensure the dataset is loaded
    if global_dataset is None:
        load_dataset()

    # Access inputs and targets from the global dataset
    inputs = global_dataset["inputs"]
    targets = global_dataset["targets"]

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
                         validation_data=validation_batches,
                         )
    return model, hist_obj


def evaluate_model(model, test_batches):
    """
    Evaluate a trained model.
    """
    # Evaluate model
    score = model.evaluate(test_batches, verbose=1)
    print(f'Test loss: {score}')


def show_training_progress(hist_obj):
    config = model_configuration()
    loss = hist_obj.history['loss']
    val_loss = hist_obj.history['val_loss']
    epochs = range(1,len(loss)+1)

    try:
        # Try to access and plot loss per iteration
        with open("trained_models\\train_losses", "rb") as fp:
            train_loss_per_batch = pickle.load(fp)
        iters = range(1, len(train_loss_per_batch) + 1)
        plt.plot(iters, np.log(train_loss_per_batch),'b', label='Training Loss')
    except:
        print("An exception occurred")

    plt.plot(epochs * config["steps_per_epoch"], np.log(loss), 'bo', label='Training Loss per epoch')
    plt.plot(epochs * config["steps_per_epoch"], np.log(val_loss), 'ro', label='Validation Loss per epoch')
    plt.legend(loc='upper right')
    plt.ylabel('Log loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


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
    trained_resnet, history = train_model(resnet, train_batches, validation_batches)

    show_training_progress(history)

    # save trained network
    #trained_resnet.save_weights('trained_net.weights.h5')

    # Evaluate trained ResNet model post training
    evaluate_model(trained_resnet, test_batches)


if __name__ == "__main__":
    training_process()