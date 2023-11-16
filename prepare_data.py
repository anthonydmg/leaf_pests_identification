import os
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
import tempfile

tqdm.pandas()


ROOT_DATA = "./Tomato"


images_files = glob(ROOT_DATA + "/**/*.jpg", recursive = True)

image_df = pd.DataFrame(images_files, columns=["image_path"])
image_df["image_root"] = image_df.image_path.apply(os.path.dirname)
image_df["image_name"] = image_df.image_path.apply(os.path.basename)
image_df["class_name"] = image_df.image_root.apply(lambda path: os.path.split(path)[-1])

X_images = image_df["image_path"]
y = image_df["class_name"]
train_df, test_df = train_test_split(image_df, test_size= 0.25, stratify=y, random_state=42, shuffle= True)

train_df["output_path"] = train_df.image_path.str.replace(ROOT_DATA, "./train", regex = False)

#os.makedirs(os.path.dirname(train_df["output_path"][0]), exist_ok= True)

test_df["output_path"] = test_df.image_path.str.replace(ROOT_DATA, "./test", regex = False)

#os.makedirs(os.path.dirname(test_df["output_path"][0]), exist_ok= True)    


def move_images(row):
    #image_path = row["image_path"]
    #print("image_path:", row["image_path"])
    im = Image.open(row["image_path"])
    os.makedirs(os.path.dirname(row["output_path"]), exist_ok= True)
    im.save(row["output_path"])

print("Guarda imagenes de entrenamiento:\n")
train_df.progress_apply(lambda row : move_images(row), axis = 1)

print("Guarda imagenes de prueba:\n")
test_df.progress_apply(lambda row : move_images(row), axis = 1)

#def save_images_subsets(image_df):
#    Image.open()

