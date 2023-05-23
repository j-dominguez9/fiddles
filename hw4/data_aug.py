from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os
from PIL import Image

datagen = ImageDataGenerator(
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.5, 1.5),
)

image_directory = r"./images/"
SIZE = 28
dataset = []
my_images = os.listdir(image_directory)
for i, image_name in enumerate(my_images):
    if image_name.split(".")[1] == "png":
        image = io.imread(image_directory + image_name)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        dataset.append(np.array(image))
x = np.array(dataset)
i = 0
for batch in datagen.flow(
    x,
    batch_size=16,
    save_to_dir=r"./augmented-images",
    save_prefix="dr",
    save_format="png",
):
    i += 1
    if i > 50:
        break
