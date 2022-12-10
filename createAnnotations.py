import os
from scipy.io import loadmat
import numpy as np

dir = os.getcwd() + "/dataset/Airplanes_Side_2"

index = 0
with open(os.getcwd() + "/dataset/airplanes.csv", "w") as csv_file:
    for filename in os.listdir(dir):
        index += 1
        try:
            mat = loadmat(dir + "/" + filename)
            coords = [x for x in mat["box_coord"][0]]
            annotationIndex = filename.split("_")[1].split(".")[0]
            annotationTitle = f"image_{annotationIndex}.jpg"
            coords.append(annotationTitle)
            vals = np.asarray(coords)
            line = f"{vals[0]}, {vals[1]}, {vals[2]}, {vals[3]}, {vals[4]}\n"
            csv_file.write(line)

        except:
            print("no file for " + dir + "/" + filename)
            continue
