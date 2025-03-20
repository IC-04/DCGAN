# DCGAN on CelebA Dataset

This repository presents an advanced implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) tailored for the CelebA dataset. The design leverages state-of-the-art deep learning methodologies within the TensorFlow ecosystem, ensuring robust convergence characteristics and superior generative quality.

---

## Table of Contents

- [Overview](#overview)
- [Technical Architecture](#technical-architecture)
  - [Discriminator Network](#discriminator-network)
  - [Generator Network](#generator-network)
- [Data Ingestion and Preprocessing](#data-ingestion-and-preprocessing)
- [Training Paradigm](#training-paradigm)
- [Repository Structure](#repository-structure)
- [Environment Setup and Dependencies](#environment-setup-and-dependencies)
- [Execution Guidelines](#execution-guidelines)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The implementation encapsulates a DCGAN variant designed to model the complex distribution of high-dimensional face imagery present in the CelebA dataset. The system is engineered with a modular architecture, facilitating experimental modifications and hyperparameter tuning. Key enhancements include:
- Dynamic GPU memory allocation to optimize computational throughput.
- Progressive feature synthesis in the generator network via transposed convolutional operations.
- Robust adversarial training using binary cross-entropy loss, optimized through the Adam optimizer with meticulously tuned learning rates.

---

## Technical Architecture

### Discriminator Network

The discriminator is a convolutional classifier meticulously architected to perform binary discrimination between real and synthesized images. It employs:
- **Convolutional Layers:** Hierarchically extract multi-scale features with kernel sizes optimized for spatial resolution reduction.
- **Leaky ReLU Activations:** Introduce controlled non-linearities to mitigate the dying ReLU phenomenon and ensure gradient propagation.
- **Dropout Regularization:** Attenuates overfitting by probabilistically deactivating neuron activations in the flattened layer.
- **Sigmoid Output:** Facilitates probabilistic interpretation for binary classification.

### Generator Network

The generator is architected to synthesize high-fidelity images from latent vectors by progressively upsampling a low-dimensional noise distribution. Its architecture includes:
- **Dense Projection:** Transforms the latent vector into a high-dimensional tensor.
- **Reshaping Mechanism:** Reconfigures the dense projection into a 3D tensor amenable to convolutional processing.
- **Transposed Convolutional Layers:** Enable learned upsampling, effectively expanding the spatial dimensions of the tensor while synthesizing realistic image features.
- **Leaky ReLU Activations:** Promote stable gradient flow during the backpropagation phase.
- **Sigmoid Output Layer:** Ensures the generated image pixels are normalized within the [0, 1] interval.

---

## Data Ingestion and Preprocessing

The CelebA dataset is acquired via TensorFlow Datasets (TFDS), and preprocessing is executed with rigorous transformations:
- **Resizing:** All images are normalized to a 64×64 spatial resolution.
- **Normalization:** Pixel intensities are scaled to a [0, 1] range using floating point precision.
- **Shuffling and Batching:** A stochastic mini-batch selection mechanism is employed to ensure that the gradients computed during the adversarial training are robust and free of bias.

---

## Training Paradigm

The adversarial training loop is engineered to iteratively update the generator and discriminator in a synchronized fashion:
- **Dual Optimization Cycle:** Employs TensorFlow's `GradientTape` to compute and apply gradients for both networks.
- **Loss Functions:** Utilizes binary cross-entropy to quantitatively evaluate the divergence between real and generated data distributions.
- **Latent Space Sampling:** Random noise vectors are sampled from a normal distribution to stimulate the generative process.
- **Checkpointing Mechanism:** Periodic saving of synthesized outputs provides real-time qualitative feedback on the generator's evolution.

---

## Repository Structure

- **`config.py`**  
  Centralizes the configuration parameters, including GPU memory growth settings and hyperparameters such as batch size, latent dimension, and number of epochs.

- **`data_loader.py`**  
  Contains the dataset pipeline that handles data ingestion, preprocessing, shuffling, and batching operations from the CelebA dataset.

- **`models.py`**  
  Defines the network architectures for both the generator and discriminator, implemented using TensorFlow Keras' Sequential API.

- **`train.py`**  
  Implements the adversarial training loop, integrating dynamic gradient computation, optimization, and checkpointing of generated images.

- **`generated_images/`**  
  Directory designated for persisting the generated samples during training for qualitative assessment.

---

## Environment Setup and Dependencies

The project requires a Python environment with the following dependencies:
- **Python:** Version 3.7 or higher
- **TensorFlow:** 2.x
- **Keras:** Integrated within TensorFlow
- **TensorFlow Datasets (TFDS)**
- **tqdm:** For progress monitoring
- **NumPy & Matplotlib:** For numerical processing and visualization

Installation can be performed using pip:

```bash
pip install tensorflow tensorflow-datasets tqdm numpy matplotlib

```
---
### Execution Guidelines

To commence training and observe the generative performance:
Clone the repository:
```bash
git clone https://github.com/your_username/dcgan-celeba.git
cd dcgan-celeba
```
Execute the training module:
```bash
python train.py
```
---
The training script orchestrates the data flow, model updates, and periodically outputs synthetic images to the generated_images/ folder.
