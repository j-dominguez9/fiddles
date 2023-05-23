Image Processing and Transfer Learning--HW4
Machine Learning 2
Joaquin Dominguez

Base Model(base.py):
Base model was created and trained using the entire MNIST dataset with 25 epochs. Validation accuracy floated around 85% (Using CPU rather than GPU).

Data Collection:
1. Letters A-G were handwritten on a white piece of paper and scanned.
2. New image was created for each letter cut from scanned image.
3. Using GIMP, each image was put into grayscale mode to reduce any color from scanner, color inverted, and clarified.*
* Clarified = any pixels that weren't supposed to be a certain color were changed (e.g., background was made entirely black and pixels inside of number were made entirely white.

Data Pre-processing(pre-data.py):
Since MNIST dataset uses certain image data configurations, the new layer would also abide by those same configurations.
1. All images fit into a 20X20 pixel box and are then centered in a 28X28 image using center_of_mass and shifting the image in the given direction.
Code for this step was adapted from : https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
2. Processed images are saved into a directory ('resized_images')

Data Augmentation(data_aug.py):
Since our dataset is really small, offline data augmentation methods were used to create 51 extra images for each letter, giving us 52 images for each letter and (52X5) 260 total images in the dataset.
1. New images were created with variations of the original image by manipulating via rotation, shear_range, flipping, brightness, and scaling.
Code for this step was adapted from : https://www.analyticsvidhya.com/blog/2021/06/offline-data-augmentation-for-multiple-images/
2. Augmented images were saved into 'augmented-images' directory and subsequently put into correct classification name folder of 'resized_images' for proper loading with tensorflow function.

Transfer Learning(join.py):
This script:
- loads the base model that was trained on the entire MNIST dataset
- loads the new dataset (A-G) with 260 images into train/test
- normalizes new data
- makes output layer of base model trainable
- creates new classification and output layers for new data
- compiles and trains neural network with validation accuracy floating around 80%.
