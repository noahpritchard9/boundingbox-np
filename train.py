import config
import tensorflow as tf
from keras.applications import VGG16
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

import cv2
import os

print("loading dataset...")
rows = open(config.ANNOTATIONS_PATH).read().strip().split("\n")

data = []
targets = []
filenames = []

for row in rows:
    row = row.split(",")
    (startX, startY, endX, endY, filename) = row
    imagePath = os.path.sep.join([config.IMAGES_PATH, filename.strip()])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h
    image = tf.keras.preprocessing.image.load_img(imagePath, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)

    data.append(image)
    targets.append((startX, startY, endX, endY))
    filenames.append(filename)

data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

print("saving testing filenames...")
f = open(config.TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()

vgg = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=tf.keras.layers.Input(shape=(224, 224, 3)),
)

vgg.trainable = False

flatten = vgg.output

flatten = tf.keras.layers.Flatten()(flatten)

bboxHead = tf.keras.layers.Dense(128, activation="relu")(flatten)
bboxHead = tf.keras.layers.Dense(64, activation="relu")(bboxHead)
bboxHead = tf.keras.layers.Dense(32, activation="relu")(bboxHead)
bboxHead = tf.keras.layers.Dense(4, activation="sigmoid")(bboxHead)

model = tf.keras.models.Model(inputs=vgg.input, outputs=bboxHead)  # not sure ab this

opt = tf.keras.optimizers.Adam(learning_rate=config.INIT_LR)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

print("training bounding box regressor...")
H = model.fit(
    trainImages,
    trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=config.BATCH_SIZE,
    epochs=config.NUM_EPOCHS,
    verbose="auto",
)

print("saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")

N = config.NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding box regression loss on training set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
