import os
import tensorflow as tf
from tqdm import tqdm
from config import LATENT_DIM, EPOCHS
from data_loader import get_dataset
from models import build_generator, build_discriminator

discriminator = build_discriminator()
generator = build_generator()
opt_gen = tf.keras.optimizers.Adam(1e-4)
opt_disc = tf.keras.optimizers.Adam(1e-4)
loss_fn = tf.keras.losses.BinaryCrossentropy()

dataset = get_dataset()
for epoch in range(EPOCHS):
    for idx, real in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        current_batch = real.shape[0]
        random_latent_vectors = tf.random.normal(shape=(current_batch, LATENT_DIM))
        fake = generator(random_latent_vectors)

        if idx % 100 == 0:
            img = tf.keras.preprocessing.image.array_to_img(fake[0])
            os.makedirs("generated_images", exist_ok=True)
            img.save(f"generated_images/generated_img_epoch{epoch:03d}_batch{idx:03d}.png")

        with tf.GradientTape() as disc_tape:
            real_predictions = discriminator(real)
            fake_predictions = discriminator(fake)
            loss_disc_real = loss_fn(tf.ones((current_batch, 1)), real_predictions)
            loss_disc_fake = loss_fn(tf.zeros((current_batch, 1)), fake_predictions)
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
        grads_disc = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)
            predictions = discriminator(fake)
            loss_gen = loss_fn(tf.ones((current_batch, 1)), predictions)
        grads_gen = gen_tape.gradient(loss_gen, generator.trainable_weights)
        opt_gen.apply_gradients(zip(grads_gen, generator.trainable_weights))
