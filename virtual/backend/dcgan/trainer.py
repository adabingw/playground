import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow.keras import layers
import time

from IPython import display

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 200
NOISE_DIM = 100
EXAMPLES_TO_GENERATE = 16

GEN_CHECKPOINT = './checkpoints/gen/generator'
DIS_CHECKPOINT = './checkpoints/dis/discriminator'

seed = tf.random.normal([EXAMPLES_TO_GENERATE, NOISE_DIM])
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def get_data(): 
    (train_images, train_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
    a = train_images.shape
    print(train_images.shape)
    train_images = train_images.reshape(a[0], a[1], a[2], a[3]).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset, a

def get_generator(shape): 
    # generator uses tf.keras.layers.Conv2DTranspose (upsampling) layers 
    # to produce an image from a seed (random noise).

    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, shape[1], shape[2], shape[3])

    return model

def get_discriminator(shape): 
    model = tf.keras.Sequential()
    # normal
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same',
                                     input_shape=[shape[1], shape[2], shape[3]]))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images, generator, discriminator):
    print("training step")
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    print("a")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    print("!")
    
def train(dataset, epochs, generator, discriminator):
    for epoch in range(epochs):
        print("epoch ", epoch)
        print(len(dataset))
        i = 1
        for image_batch in dataset:
            print(i)
            train_step(image_batch, generator, discriminator)
            i += 1
        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)
    
def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        # plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='grey')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

def main(): 
    train_dataset, shape = get_data() 
    generator = get_generator(shape) 
    discriminator = get_discriminator(shape) 
    
    print("start training...")
    train(train_dataset, EPOCHS, generator, discriminator)
    print("training done!")
        
    generator.save_weights(GEN_CHECKPOINT) 
    discriminator.save_weights(DIS_CHECKPOINT) 
    
    def display_image(epoch_no):
      return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

    display_image(EPOCHS)
    
def output(): 
    print("GENERATING OUTPUT")
    shape = (50000, 32, 32, 3)
    generator = get_generator(shape)
    discriminator = get_discriminator(shape) 
    
    generator.load_weights(GEN_CHECKPOINT) 
    discriminator.load_weights(DIS_CHECKPOINT) 
    
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0])
    plt.show()
    print("...")

if __name__ == "__main__":
    output()    