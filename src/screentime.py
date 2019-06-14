import sys
import cv2
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append("../utils")
import detections
from tracking import IDTracker
from mtcnn_detector import MtcnnDetector


# Set model files
detector_file = "../models/mtcnn"
facenet_frozen_model = "../models/facenet/facenet.pb"
tf.logging.info("Loading face detector models...")

# Load facenet
detections.load_model(facenet_frozen_model)

# Load Mtcnn detector
ctx = "gpu" if tf.test.is_gpu_available() else "cpu"
tf.logging.log_if(tf.logging.info, "Running on GPU", ctx == "gpu")
mtcnn_detector = MtcnnDetector(
    model_folder=detector_file, num_worker=4,
    accurate_landmark=True, ctx=ctx)


class TrackableFace(object):

    def __init__(self, _id):
        self.objectID = _id
        self.is_counted = False
        self.username = None


def main(args):
    # Load video either from webcam or video file
    video_file = 0 if args.video == "0" else args.video
    vcap = cv2.VideoCapture(video_file)

    # Load embeddings  and user metadatata assign usernames
    embeds = np.load(args.embeddings)
    user_data = pd.read_csv(args.user_data)

    # Detect, verificate and track faces
    track_screentime(args.tracker, vcap, args.interval,
                     embeds, user_data, write_path=args.output)


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
    if not to.is_counted:
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


def detect(image, minSize=70, tol=0.95):
    detections = mtcnn_detector.detect_face(image)
    if detections is None:
        return []
    else:
        detections = detections[0]
    bboxes = []
    for i in range(len(detections)):
        # Filter by level of confidence
        confidence = detections[i, -1]
        if confidence >= tol:
            x1 = int(detections[i, 0])
            y1 = int(detections[i, 1])
            x2 = int(detections[i, 2])
            y2 = int(detections[i, 3])
            w = x2 - x1 + 15
            h = y2 - y1 + 15
            if w > minSize:
                bboxes.append((x1, y1, w, h))
    return bboxes


def track_screentime(tracker, vcap, n, embeddings, user_data, write_path=None):
    # Setup screentime of each user
    users = user_data.name.values
    stime = dict(zip(users, np.tile(0, len(users))))
    stime["unassigned"] = 0
    # Initialize variables
    frame_counter = 0
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    # Save video
    if write_path is not None:
        width = int(vcap.get(3))
        height = int(vcap.get(4))
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        writer = cv2.VideoWriter(write_path, fourcc, fps, (width, height))
    # Track recorded ids
    trackableFaces = {}
    while True:
        ret, frame = vcap.read()
        # There was an error opening file
        if not ret:
            tf.logging.error("Could not read from video feed")
            break

        # DO: Tracking
        if frame_counter == 0:
            # status = "Initializing"
            multitracker = IDTracker(tracker, maxDetections=5)
            face_detections = detect(frame)
            for coord in face_detections:
                multitracker.register(frame, coord)
            face_detections = multitracker.data["detections"]
        # Do face detection
        elif frame_counter % n == 0 and frame_counter != 0:
            # status = "Detecting"
            face_detections = detect(frame)
            face_detections = multitracker.assign_ids(frame, face_detections)
        # Update using chosen tracker
        else:
            # status = "Tracking"
            face_detections = multitracker.update(frame)

        # DO: bounding boxes and screentime
        for _id, (x, y, w, h) in face_detections.items():
            # Do face_verification
            trackableFaces = assign_username(
                frame, trackableFaces, _id, (x, y, w, h), embeddings, user_data)
            # Write user data
            username = trackableFaces.get(_id).username
            color = (0, 0, 255) if username == "unassigned" else (0, 255, 0)
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # WRITE: username
            cv2.putText(
                frame, username, (x, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # WRITE: screentime
            stime[username] += 1 / fps
            text = "time: {}s".format(np.round(stime[username], 1))
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Delete object if it is not found
            if multitracker.data.loc[_id]["failures"] >= multitracker.maxDetections:
                multitracker.deregister(_id)
                del trackableFaces[_id]

        cv2.imshow("face_detections", frame)
        if write_path is not None:
            writer.write(frame)
        if cv2.waitKey(fps) & 0xFF == ord("q"):
            tf.logging.info("Finished video processing")
            break
        frame_counter += 1
    # Close everything
    vcap.release()
    if write_path is not None:
        writer.release()
        tf.logging.info("Video is annotated in {}".format(write_path))
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("video", type=str,
                        help="Video on which to detect faces")
    parser.add_argument("user_data", type=str,
                        help="file containing user metadata")
    parser.add_argument("embeddings", type=str,
                        help="file containing user embeddings")
    parser.add_argument("--tracker", "-t", type=str, default="kcf",
                        help="opencv tracking algorithm")
    parser.add_argument("--interval", "-n", type=int, default=10,
                        help="frame interval for detection")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="write video to output path")

    return parser.parse_args(argv)


if __name__ == "__main__":
    # Allow logs for debugging
    tf.logging.set_verbosity(tf.logging.INFO)
    main(parse_arguments(sys.argv[1:]))
