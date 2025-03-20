import tensorflow_datasets as tfds
import tensorflow as tf
from config import BATCH_SIZE

def preprocess(sample):
    image = sample["image"]
    image = tf.image.resize(image, (64, 64))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def get_dataset():
    (ds_train,), ds_info = tfds.load("celeb_a", split=["train"], with_info=True, as_supervised=False)
    dataset = ds_train.map(preprocess)
    dataset = dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE)
    return dataset
