import cv2
import numpy as np
import tensorflow as tf


# TODO: verification and clustering using Facenet
def load_model(frozen_model, graph=tf.get_default_graph()):
    with graph.as_default():
        tf.logging.info("Loading Facenet embedding model...")
        graph_def = tf.GraphDef()
        f = tf.gfile.GFile(frozen_model, "rb")
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name="")


def align_data(images, bboxes):
    img_list = []
    for img, bbox in zip(images, bboxes):
        cropped = crop(img, bbox)
        aligned = align(cropped)
        prewhitened = prewhiten(aligned)
        img_list.append(prewhitened)
    return np.stack(img_list)


def create_embeddings(images, graph=tf.get_default_graph()):
    # Convert to batch format
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
    with tf.Session() as sess:
        # Get input and output tensors
        images_placeholder = graph.get_tensor_by_name("input:0")
        embeddings = graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        tf.logging.info("Creating embeddings...")
        emb = sess.run(embeddings, feed_dict=feed_dict)
    return emb


def crop(img, bbox):
    x, y, w, h = bbox
    cropped = img[int(y): int(y + h), int(x): int(x + w)]
    return cropped


def prewhiten(x):
    mu = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = (x - mu) / std_adj
    return y


def align(img):
    return cv2.resize(img, (160, 160))


def write_graph(summary_file, graph=tf.get_default_graph()):
    tf.logging.info("Writing graph to {}".format(summary_file))
    writer = tf.summary.FileWriter(logdir=summary_file, graph=graph)
    writer.flush()
