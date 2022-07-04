import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib
import matplotlib.pyplot as plt

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
# Let's visualize an image
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))

batch_size = 32
img_height = 180
img_width = 180
# It's good practice to use a validation split when developing your model. 
# Here I will use 80% of the images for training and 20% for validation.
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
# You can find the class names in the `class_names` attribute on these datasets.

class_names = train_ds.class_names
print(class_names)
### Visualize the data

# Here are the first nine images from the training dataset.

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
# You can train a model using these datasets by passing them to model.fit (shown later in this tutorial). 
# If you like, you can also manually iterate over the dataset and retrieve batches of images:
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break
# The image_batch is a tensor of the shape (32, 180, 180, 3). 
# This is a batch of 32 images of shape 180x180x3 (the last dimension refers to color channels RGB). 
# The label_batch is a tensor of the shape (32,), 
# these are corresponding labels to the 32 images.
# You can call .numpy() on either of these tensors to convert them to a numpy.ndarray.
# Standardize the data
normalization_layer = tf.keras.layers.Rescaling(1./255)
# There are two ways to use this layer. You can apply it to the dataset by calling `Dataset.map`:
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

# Or, you can include the layer inside 
# your model definition to simplify deployment. 
# You will use the second approach here.

# Note: If you would like to scale pixel values to `[-1,1]` 
# you can instead write `tf.keras.layers.Rescaling(1./127.5, offset=-1)`

# Note: You previously resized images using the `image_size` argument of `tf.keras.utils.image_dataset_from_directory`. 
# If you want to include the resizing logic in your model as well, 
# you can use the `tf.keras.layers.Resizing` layer.

# Configure the dataset for performance
# Let's make sure to use buffered prefetching so you can yield data from disk without having I/O become blocking. These are two important methods you should use when loading data:

# Dataset.cache keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
# Dataset.prefetch overlaps data preprocessing and model execution while training.



AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)





