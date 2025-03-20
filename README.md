DCGAN Implementation for CelebA Dataset

This repository contains a TensorFlow/Keras implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) for synthesizing face images using the CelebA dataset. Two separate scripts, main.py and train.py, encapsulate similar training pipelines with minor discrepancies. Both scripts are engineered to execute in a GPU-enabled environment with dynamic memory allocation.

Environment and Dependencies

TensorFlow 2.x: Utilized for high-level model definition and gradient-based optimization.
Keras API: Employed for constructing Sequential models for both the discriminator and generator.
NumPy: Used for numerical operations and random number generation.
Matplotlib: Utilized for saving generated image outputs.
TQDM: Integrated for progress tracking during iterative training loops.
Installation:

pip install tensorflow numpy matplotlib tqdm
Dataset Preprocessing

The data ingestion pipeline leverages keras.preprocessing.image_dataset_from_directory to ingest the dataset from a folder (celeb_dataset). Images are resized to a fixed resolution of 64×64 and normalized to a [0, 1] range by scaling the pixel values (i.e., division by 255). The mapping function ensures that the normalization is applied to all batches in the pipeline.

Model Architectures

Discriminator
Input: Images of shape (64, 64, 3).
Convolutional Layers: A series of convolutional layers with kernel size 4, stride 2, and ‘same’ padding to progressively downsample the spatial dimensions.
Activation: LeakyReLU with an alpha of 0.2, providing non-linearity while mitigating vanishing gradients.
Regularization: A Dropout layer (rate = 0.2) is used post-flattening to mitigate overfitting.
Output Layer: A Dense layer with sigmoid activation to output a scalar probability, representing the likelihood that the input image is real.
The discriminator is optimized using the Binary Crossentropy loss function, targeting a maximization of log-likelihood for real images and minimization for fake images.

Generator
Input: A latent space vector of dimensionality 128.
Dense & Reshape: Initial fully connected layer projects the latent vector into a high-dimensional space, reshaped into a tensor of shape (8, 8, 128).
Deconvolution: Multiple Conv2DTranspose layers are used to upscale the feature maps, each followed by a LeakyReLU activation. The use of transposed convolutions allows for learnable upsampling.
Final Layer: A Conv2D layer with a kernel size of 5 and sigmoid activation maps the features to an output image with 3 channels. The sigmoid activation ensures the pixel intensities are bounded between 0 and 1.
Both models are instantiated as Sequential models, and their summaries are printed to verify layer connectivity and parameter counts.

Training Loop and Optimization Strategy

Both scripts implement a standard GAN training loop that alternates updates between the discriminator and the generator:

Latent Vector Sampling: A batch-wise random normal distribution is sampled to generate latent vectors.
Forward Pass:
The generator produces a batch of synthetic images.
The discriminator processes both real and generated images.
Loss Computation:
Discriminator Loss: Computed as the average of the binary crossentropy losses for real (label = 1) and fake (label = 0) images.
Generator Loss: Computed by evaluating the discriminator’s output on the generated images with target labels set to 1. This is equivalent to maximizing the probability of generated images being classified as real.
Backpropagation: Gradients are calculated via tf.GradientTape for both models, and the Adam optimizer (learning rate = 1e-4) is applied to update model parameters.
Checkpointing: Periodically, a generated sample image is saved to disk for visual inspection. Notably, the output directory nomenclature differs between main.py (gen_images) and train.py (generated_images).
Code Specifics and Considerations

GPU Memory Management: The code manually sets memory growth on the first available GPU to allow for dynamic memory allocation. It is recommended to check for GPU availability to prevent runtime errors on CPU-only systems.
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
Batch Processing: Both implementations iterate over the dataset in mini-batches, which is crucial for stochastic gradient descent optimization and ensuring efficient GPU utilization.
File I/O: Generated images are saved periodically (every 100 batches) to provide an operational checkpoint. Consistency in directory naming conventions between the two scripts is recommended for unambiguous data management.
Code Duplication: There is significant overlap between main.py and train.py. Consider refactoring common components into modular functions or a shared library to improve maintainability.
Execution

To run the training loop, execute either of the scripts as required:

main.py:
python main.py
Output: Saves images in the gen_images directory.
train.py:
python train.py
Output: Saves images in the generated_images directory.
Ensure that the directory structure contains the celeb_dataset folder with properly formatted image data.

Conclusion

This implementation leverages standard practices in adversarial training and deep convolutional architectures. The technical rigor applied in defining network layers, optimization hyperparameters, and training protocols adheres to contemporary research standards for generative modeling. Consolidation and further modularization of duplicate code segments are recommended for enhanced scalability and ease of maintenance.

