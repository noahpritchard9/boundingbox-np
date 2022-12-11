import config
import tensorflow as tf
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

# add argument parsing so we can pass in a file or directory
# to run the model on
# Example: python3 predict.py --input image.jpg
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--input",
    required=True,
    help="path to input image/text file of image filenames",
)

# if filetype is text file, we need to loop through all images
args = vars(ap.parse_args())

filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

if filetype == "text/plain":
    filenames = open(args["input"]).read().strip().split("\n")
    imagePaths = []

    for f in filenames:
        p = os.path.sep.join([config.IMAGES_PATH, f])
        imagePaths.append(p)

# load the model from memory
print("loading image detector...")
model = tf.keras.models.load_model(config.MODEL_PATH)

# load each image, preprocess/scale them as our model expects
# this means they must match our training data
for ip in imagePaths:
    image = tf.keras.preprocessing.image.load_img(ip, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds

    image = cv2.imread(ip)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # get the predicted coordinates and plot them on top of the image
    # when this file is executed, it will open an output dialogue
    # showing the image and the bounding box

    # if you pass in a single file, press any key to end the program
    # if you pass in a text file containing multiple files,
    # pressing any key will move you to the next image
    # until all images have been seen, then the program will end
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)
