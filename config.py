import os
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set memory growth for GPUs
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Hyperparameters
BATCH_SIZE = 32
LATENT_DIM = 128
EPOCHS = 10
