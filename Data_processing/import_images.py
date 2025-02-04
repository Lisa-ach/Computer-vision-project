import pandas as pd
import os
import cv2
import numpy as np

def load_images_cv2(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

<<<<<<< HEAD:Computer vision project/import_images.py

=======
>>>>>>> 2d8e75cb70e670acf6202b72a61b935623e21c43:Computer vision project/Data_processing/import_images.py
images_normal = load_images_cv2("Data/normal/")
images_potholes = load_images_cv2("Data/potholes/")

images = images_normal + images_potholes
Y = [0]*len(images_normal) +[1]*len(images_potholes)