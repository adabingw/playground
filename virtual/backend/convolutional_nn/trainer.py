import numpy as np
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pathlib
import sys
import logging

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

tf.get_logger().setLevel('ERROR')

batch_size = 32
img_height = 180
img_width = 180
class_names = ['sunflowers', 'tulips']

def init_model(class_names, img_height, img_width): 
    num_classes = len(class_names)
    
    data_augmentation = Sequential([
        layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, name="outputs")
    ])
    
    return model 

def main(): 
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    print("AAAAAAAAAA ", data_dir)
    train_ds = tf.keras.utils.image_dataset_from_directory(
                    data_dir,
                    validation_split=0.2,
                    subset="training",
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
                    data_dir,
                    validation_split=0.2,
                    subset="validation",
                    seed=123,
                    image_size=(img_height, img_width),
                    batch_size=batch_size)
    print("data download finished")
    class_names = train_ds.class_names
    
    # configure dataset for performance 
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    print("configured dataset \n")
    
    train(train_ds, val_ds)
    

def train(train_ds, val_ds): 
    model = init_model(class_names, img_height=img_height, img_width=img_width)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    epochs=10
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save('cnn')
    
def classify(url): 
    model = keras.models.load_model('cnn')
    path = tf.keras.utils.get_file('data', origin=url)
    img = tf.keras.utils.load_img(path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    
if __name__ == "__main__":
    # print(sys.argv)
    if len(sys.argv) == 1: 
        print("calling main to train model") 
        main() 
    elif len(sys.argv) == 3: 
        print("classifying image from url ", sys.argv[-1]) 
        print("\n")
        classify(sys.argv[-1])