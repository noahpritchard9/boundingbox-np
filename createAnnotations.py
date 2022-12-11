import os
from scipy.io import loadmat
import numpy as np

# This file is used to translate the data in the matlab files
# into bounding box coordinates that I can use to train my model

# First get the current path, then for each matlab file,
# read in the coordinates and format them into a csv line.
# Add in the image name corresponding to the coordinates,
# then write each csv line into the annotations file (airplanes.csv)

dir = os.getcwd() + "/dataset/Airplanes_Side_2"

with open(os.getcwd() + "/dataset/airplanes.csv", "w") as csv_file:
    for filename in os.listdir(dir):
        try:
            mat = loadmat(dir + "/" + filename)
            coords = [x for x in mat["box_coord"][0]]
            annotationIndex = filename.split("_")[1].split(".")[0]
            annotationTitle = f"image_{annotationIndex}.jpg"
            coords.append(annotationTitle)
            vals = np.asarray(coords)
            line = f"{vals[2]},{vals[0]},{vals[3]},{vals[1]},{vals[4]}\n"
            csv_file.write(line)

        except:
            print("no file for " + dir + "/" + filename)
            continue
