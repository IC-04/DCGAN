# DCGAN Implementation for CelebA

This repository contains a TensorFlow/Keras implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) designed to synthesize face images from the CelebA dataset. The project comprises two separate scripts (`main.py` and `train.py`), each encapsulating a nearly identical training pipeline with minor discrepancies. This implementation is optimized for execution on GPU-enabled systems with dynamic memory management.

---

## Environment and Dependencies

- **TensorFlow 2.x** – Core framework for building and training the neural networks.
- **Keras API** – Utilized for defining the sequential models for both the generator and discriminator.
- **NumPy** – For numerical computations and random latent vector generation.
- **Matplotlib** – Employed for saving generated images.
- **TQDM** – For progress monitoring during training iterations.

**Installation:**

```bash
pip install tensorflow numpy matplotlib tqdm
