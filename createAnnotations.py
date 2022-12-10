import os
from scipy.io import loadmat
import csv
import pandas
import numpy as np

dir = os.getcwd() + "/dataset/Airplanes_Side_2"

index = 0
with open(os.getcwd() + "/dataset/airplanes.csv", "w") as csv_file:
    writer = csv.writer(csv_file)
    for filename in os.listdir(dir):
        index += 1
        try:
            mat = loadmat(dir + "/" + filename)
            bb_list: list[list[str]] = [
                [element for element in upperElement]
                for upperElement in mat["box_coord"]
            ]
            annotationIndex = filename.split("_")[1].split(".")[0]
            annotationTitle = f"image_{annotationIndex}.jpg"
            bb_list[0].append(annotationTitle)
            vals = np.asarray(bb_list)
            pandas.DataFrame(vals).to_csv()

            writer.writerow(vals)

        except:
            print("no file for " + dir + "/" + filename)
            continue
