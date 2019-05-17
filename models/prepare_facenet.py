import os
import zipfile
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str,
                    help="Path to facenet zip file")
args = parser.parse_args()
model_path = os.path.expanduser(args.model_path)

# Unzip
zip_ref = zipfile.ZipFile(model_path, "r")
zip_ref.extractall(".")
zip_ref.close()

# Rename dir and files
MODEL_NAME = "facenet"
dir_name = os.path.splitext(model_path.split("/")[-1])[0]
os.rename(dir_name, MODEL_NAME)
file_paths = [os.path.join(MODEL_NAME, file) for file in os.listdir(MODEL_NAME)]
filenames = [os.path.splitext(fn)[0] for fn in os.listdir(MODEL_NAME)]

for fn, file_path in zip(filenames, file_paths):
    renamed = file_path.replace(fn, MODEL_NAME)
    os.rename(file_path, renamed)
