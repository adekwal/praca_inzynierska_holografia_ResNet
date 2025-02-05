import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, GlobalAveragePooling2D, \
    Conv2D, Lambda, Input, BatchNormalization, Activation, MaxPool2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import shutil
import h5py
import pickle

dataset_file = r"C:/Users/Monika Walocha/Desktop/adek files/_python/praca_inzynierska/dane50_compressed.h5" # set the path to training dataset file
global_dataset = None

def get_dataset_size(file_path):
    with h5py.File(file_path, 'r') as h5f:
        dataset = h5f['inputs']
        return dataset.shape[0]


def model_configuration():
    global global_dataset  # use the global dataset variable

    # Ensure the dataset is loaded
    if global_dataset is None:
        load_dataset()

    # Generic configuration
    width, height, channels = 512, 512, 1
    batch_size = 1 # number of samples being analyzed at once
    validation_split = 0.2 # 80% of the dataset will be used for training, 10% for validation and 10% for the test
    verbose = 1
    n = 3  # number of residual blocks in a single group
    init_fm_dim = 16  # initial number of feature maps; doubles as the feature map size halves
    shortcut_type = "identity"  # shortcut type: "identity" or "projection"

    num_samples = get_dataset_size(dataset_file)
    train_size = (1 - validation_split) * num_samples
    val_size = validation_split * num_samples

    # Calculate parameters
    maximum_number_iterations = 160000
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
        os.path.join(os.getcwd(), 'trained_models', 'epoch_{epoch:02d}_model_checkpoint.keras'),
        save_freq="epoch"
    )

    class SaveBatchLoss(Callback):
        def on_train_begin(self, logs={}):
            self.train_losses = []
        def on_train_batch_end(self, batch, logs={}):
            self.train_losses.append(logs.get('loss'))
        def on_train_end(self, logs={}):
            with open("trained_models\\train_losses", "wb") as fp:  # pickling
                pickle.dump(self.train_losses, fp)

    save_batch_loss = SaveBatchLoss()

    # Add callbacks to a list
    callbacks = [
        tensorboard,
        checkpoint,
        save_batch_loss
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

        global_dataset = {"inputs": h5f['inputs'], "targets": h5f['targets']} # use lazy loading to create an object

        num_samples = global_dataset['inputs'].shape[0]
        print(f"Loaded dataset: {num_samples} samples into global_dataset.")

    except FileNotFoundError:
        raise FileNotFoundError("The dataset file was not found.")
    except KeyError as e:
        raise KeyError(f"Missing expected data key in the file: {e}")
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the dataset: {e}")


def data_generator(inputs, targets, batch_size):
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
    validation_split = config["validation_split"]
    batch_size = config["batch_size"]

    # Calculate set size
    total_samples = len(inputs)
    val_test_size = int(total_samples * validation_split)
    val_size = val_test_size // 2
    train_size = total_samples - val_test_size

    # Calculate the indices for dataset splitting
    train_indices = range(0, train_size)
    val_indices = range(train_size, train_size + val_size)
    test_indices = range(train_size + val_size, total_samples)

    # Define generators
    train_gen = lambda: data_generator(inputs[train_indices], targets[train_indices], batch_size)
    val_gen = lambda: data_generator(inputs[val_indices], targets[val_indices], batch_size)
    test_gen = lambda: data_generator(inputs[test_indices], targets[test_indices], batch_size)

    # Create TensorFlow sets
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

    # Prepare datasets
    train_dataset = train_dataset.shuffle(buffer_size=1024).repeat().prefetch(tensorflow.data.AUTOTUNE)
    validation_dataset = validation_dataset.repeat().prefetch(tensorflow.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tensorflow.data.AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


def residual_block(x, number_of_filters):
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

    return x


def ResidualBlocks(x):
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

    return model, hist_obj


def evaluate_model(model, test_batches):
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
    # Get dataset
    train_batches, validation_batches, test_batches = preprocessed_dataset()

    # Prepare directory for logs
    log_dir = os.path.join(os.getcwd(), "logs")
    if os.path.exists(log_dir) and os.path.isdir(log_dir):
         shutil.rmtree(log_dir)

    # Initialize ResNet
    resnet = init_model()

    # Train ResNet model
    trained_resnet, history = train_model(resnet, train_batches, validation_batches)

    show_training_progress(history)

    # Evaluate trained ResNet model post training
    evaluate_model(trained_resnet, test_batches)


if __name__ == "__main__":
    training_process()