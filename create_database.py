import os
import sys
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append("utils")
import detections
from mtcnn_detector import MtcnnDetector


def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load facenet
    frozen_model = os.path.abspath("models/facenet/facenet.pb")
    detections.load_model(frozen_model)
    # Load mtccn model
    model_dir = "models/mtcnn"
    # Use gpu if available
    ctx = "gpu" if tf.test.is_gpu_available() else "cpu"
    tf.logging.log_if(tf.logging.INFO, "Running on GPU", ctx == "gpu")
    mtcnn_detector = MtcnnDetector(
        model_folder=model_dir, num_worker=4, ctx=ctx)
    # Infer usernames from filenames
    filenames = [
        os.path.join(args.image_dir, file) for file in os.listdir(args.image_dir)]
    names = [
        os.path.splitext(file.split("/")[-1])[0] for file in filenames]
    # Create database and save it
    create_database(mtcnn_detector, filenames, names,
                    args.user_data, args.embeddings)


def create_database(mtcnn_detector, image_paths, names, user_data, emb_file):
    """
    image_paths: list of paths to images that are named based
    on the key ID the user wishes to acquire on the face_verification
    script
    """
    emb_fn = os.path.splitext(emb_file)[0]
    # Get full path to avoid false duplicates
    image_paths = [os.path.abspath(img_path) for img_path in image_paths]
    # Extracting already created data
    if os.path.exists(user_data) and os.path.exists(emb_file):
        data = pd.read_csv(user_data)
        embs = np.load(emb_file)
        # The file might contain users that are already registered
        # in the database. Instead of creating unnecesary embeddings
        # remove the duplicates.
        duplicates = data[data.path.isin(image_paths)].values
        tf.logging.info(
            "Found {} duplicates. Removing them...".format(len(duplicates)))
        for (used_name, used_path) in duplicates:
            names.remove(used_name)
            image_paths.remove(used_path)
    # First time creating database on specified files
    else:
        data = pd.DataFrame(columns=["name", "path"])
        embs = None
    # All of the image paths were duplicates
    if len(image_paths) == 0:
        tf.logging.warn("All of the images were duplicates. Exiting...")
        return None
    # Create embeddings and track time
    aligned = load_and_align(mtcnn_detector, image_paths)
    t1 = time.time()
    embeddings = detections.create_embeddings(aligned)
    t2 = time.time()
    tf.logging.info(
        "{:0.2f} seconds for {} embeddings...".format(t2 - t1, len(embeddings)))
    # Save data
    data = data.append(
        pd.DataFrame({"name": names, "path": image_paths}), ignore_index=True)
    embs = np.vstack((embs, embeddings)) if embs is not None else embeddings
    tf.logging.info("Saving user data in {}...".format(user_data))
    data.to_csv(user_data, index=None)
    tf.logging.info("Saving embeddings in {}...".format(emb_file))
    np.save(emb_fn, embs)


def mtcnn_detect(mtcnn_detector, img):
    results = mtcnn_detector.detect_face(img)
    # Detector did not work properly
    if results is None or len(results[0]) > 1:
        tf.logging.warn("detector did not work properly. Select face manually")
        bbox = cv2.selectROI(img)
        cv2.destroyAllWindows()
        return bbox
    detection = np.squeeze(results[0].astype("int"))
    x1, y1, x2, y2, score = detection
    # Delete later
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("mtcnn", img)
    cv2.waitKey(0) & 0XFF
    cv2.destroyAllWindows()
    w, h = x2 - x1, y2 - y1
    return (x1, y1, w, h)


def load_and_align(mtcnn_detector, image_paths):
    images = []
    bboxes = []
    for path in image_paths:
        img = cv2.imread(path)
        bbox = mtcnn_detect(mtcnn_detector, img)
        images.append(img)
        bboxes.append(bbox)
    aligned = detections.align_data(images, bboxes)
    return aligned


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("user_data", type=str,
                        help="csv file containing user metadata")
    parser.add_argument("embeddings", type=str,
                        help="file containing user embeddings")
    parser.add_argument("image_dir", type=str,
                        help="directory containing images to add to database")

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
