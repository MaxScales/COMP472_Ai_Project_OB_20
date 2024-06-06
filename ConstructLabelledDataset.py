import os
from numpy import asarray
from skimage import io
import csv

script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "Train")
test_path = os.path.join(script_dir, "Test")
Combined_path_csv = os.path.join(script_dir, "CombinedDataset//Dataset.csv")

classes = ("Angry", "Engaged", "Happy", "Neutral")


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            img = io.imread(file_path)
            if img is not None:
                images.append(img)
    return images


def construct_tuples(images, label):
    labelled_images = [()]
    for image in images:
        labelled_images += (image, label)
    return labelled_images


str_Angry = 'Angry'
str_Happy = 'Happy'
str_Neutral = 'Neutral'
str_Engaged = 'Engaged'

Train_Angry_Images = load_images_from_folder(os.path.join(train_path, "Angry"))
Train_Happy_Images = load_images_from_folder(os.path.join(train_path, "Happy"))
Train_Engaged_Images = load_images_from_folder(os.path.join(train_path, "Engaged"))
Train_Neutral_Images = load_images_from_folder(os.path.join(train_path, "Neutral"))
Test_Angry_Images = load_images_from_folder(os.path.join(test_path, "Angry"))
Test_Happy_Images = load_images_from_folder(os.path.join(test_path, "Happy"))
Test_Neutral_Images = load_images_from_folder(os.path.join(test_path, "Neutral"))
Test_Engaged_Images = load_images_from_folder(os.path.join(test_path, "Engaged"))

All_Image_Touples = construct_tuples(Train_Angry_Images, str_Angry)
All_Image_Touples += construct_tuples(Train_Happy_Images, str_Happy)
All_Image_Touples += construct_tuples(Train_Engaged_Images, str_Engaged)
All_Image_Touples += construct_tuples(Train_Neutral_Images, str_Neutral)
All_Image_Touples += construct_tuples(Test_Angry_Images, str_Angry)
All_Image_Touples += construct_tuples(Test_Happy_Images, str_Happy)
All_Image_Touples += construct_tuples(Test_Engaged_Images, str_Engaged)
All_Image_Touples += construct_tuples(Test_Neutral_Images, str_Neutral)

np_array_tuple = asarray(All_Image_Touples, dtype='object')

with open(Combined_path_csv, 'w') as out:
    csv_out = csv.writer(out)
    csv_out.writerow(['Img', 'Label'])
    csv_out.writerows(All_Image_Touples)
