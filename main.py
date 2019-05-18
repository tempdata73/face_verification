import sys
import cv2
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import detections
from utils.tracking import IDTracker


class TrackableFace(object):

    def __init__(self, _id):
        self.objectID = _id
        self.is_counted = False
        self.username = None


def main(args):
    global net

    # Show logs
    tf.logging.set_verbosity(tf.logging.INFO)
    # Set model files
    detector_file = "models/opencv_detector/face_detector_uint8.pb"
    detector_config_file = "models/opencv_detector/face_detector.pbtxt"
    facenet_frozen_model = "models/facenet/facenet.pb"
    # Initialize models
    tf.logging.info("Loading face detector model...")
    net = cv2.dnn.readNetFromTensorflow(
        detector_file, detector_config_file)
    detections.load_model(facenet_frozen_model)
    # Necessary parameters
    vcap = cv2.VideoCapture(0)
    embeds = np.load(args.embeddings)
    user_data = pd.read_csv(args.user_data)
    # Detect, verificate and track faces
    track_faces(args.tracker, vcap, args.interval, embeds, user_data)


def align_detections(frame, bbox):
    cropped = detections.crop(frame, bbox)
    aligned = detections.align(cropped)
    prewhitened = detections.prewhiten(aligned)
    return prewhitened


def assign_username(frame, trackableObjects, _id, bbox,
                    embeddings, user_data, threshold=1.1):
    to = trackableObjects.get(_id, None)
    if to is None:
        to = TrackableFace(_id)
    elif not to.is_counted:
        aligned = align_detections(frame, bbox)
        user_embed = detections.create_embeddings(aligned)
        dist = np.linalg.norm(embeddings - user_embed, axis=1)
        idx, min_dist = np.argmin(dist), np.min(dist)
        if min_dist > threshold:
            to.username = "unassigned"
        else:
            to.username = user_data.loc[idx]["name"]
        to.is_counted = True
    trackableObjects[_id] = to
    return trackableObjects


def detect(image, shape):
    height, width = shape
    blob = cv2.dnn.blobFromImage(
        image, 1, (height, width), [104, 117, 123], False, False)
    net.setInput(blob)
    return net.forward()


def get_bboxes(face_detections, shape, tol=0.95):
    bboxes = []
    height, width = shape
    # Get object coordinates
    for i in range(face_detections.shape[2]):
        # Filter by level of confidence
        confidence = face_detections[0, 0, i, 2]
        if confidence >= tol:
            x1 = int(face_detections[0, 0, i, 3] * width)
            y1 = int(face_detections[0, 0, i, 4] * height)
            x2 = int(face_detections[0, 0, i, 5] * width + 45)
            y2 = int(face_detections[0, 0, i, 6] * height + 15)
            bboxes.append((x1, y1, x2, y2))
    bboxes = np.asarray(bboxes).reshape((-1, 4))
    # Object is not useful if not in image
    bboxes = np.delete(
        bboxes,
        np.where((bboxes[:, 2] > width) | (bboxes[:, 3] > height)),
        axis=0)
    # Format requires width and height, not bottom right coords
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    # Format required for object registration in tracking
    bboxes = [tuple(coords) for coords in bboxes.tolist()]
    return bboxes


def track_faces(tracker, vcap, n, embeddings, user_data):
    # Setup tracker
    frame_counter = 0
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    trackableFaces = {}
    while True:
        ret, frame = vcap.read()
        if not ret:
            tf.logging.error("Could not read from video feed")
            break
        # Initialize tracker
        if frame_counter == 0:
            # status = "Initializing"
            multitracker = IDTracker(tracker, maxDetections=10)
            face_detections = detect(frame, (height, width))
            face_detections = get_bboxes(face_detections, (height, width))
            for coord in face_detections:
                multitracker.register(frame, coord)
            face_detections = multitracker.data["detections"]
        # Do face detection
        elif frame_counter % n == 0 and frame_counter != 0:
            # status = "Detecting"
            face_detections = detect(frame, (height, width))
            face_detections = get_bboxes(face_detections, (height, width))
            face_detections = multitracker.assign_ids(frame, face_detections)
        # Update using chosen tracker
        else:
            # status = "Tracking"
            face_detections = multitracker.update(frame)
        for _id, (x, y, w, h) in face_detections.items():
            # Do face_verification
            trackableFaces = assign_username(
                frame, trackableFaces, _id, (x, y, w, h), embeddings, user_data)
            # Write user data
            username = trackableFaces.get(_id).username
            color = (0, 0, 255) if username == "unassigned" else (0, 255, 0)
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, username, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Delete object if it is not found
            if multitracker.data.loc[_id]["failures"] >= multitracker.maxDetections:
                del trackableFaces[_id]
        cv2.imshow("face_detections", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            tf.logging.info("Finished video processing")
            break
        frame_counter += 1
    # Close everything
    vcap.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("user_data", type=str,
                        help="file containing user metadata")
    parser.add_argument("embeddings", type=str,
                        help="file containing user embeddings")
    parser.add_argument("--tracker", "-t", type=str, default="kcf",
                        help="opencv tracking algorithm")
    parser.add_argument("--interval", "-n", type=int, default=20,
                        help="frame interval for detection")

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
