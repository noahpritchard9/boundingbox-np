import config
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2  # not working
import os

print("loading dataset...")
rows = open(config.ANNOTATIONS_PATH).read().strip().split("\n")

data = []
targets = []
filenames = []

for row in rows:
    row = row.split(",")
    (filename, startX, startY, endX, endY) = row
    imagePath = os.path.sep.join([config.IMAGES_PATH, filename])
    image = cv2.imread(imagePath)
