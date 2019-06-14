# Real time face tracking and verification
Opencv-tensorflow-mxnet implementation of face tracking and verification on real time using either CPU or GPU.
The pretrained models used for face verification were obtained from [David Sandberg's github repo](https://github.com/davidsandberg/facenet)
which are tensorflow implementations of [FaceNet](https://arxiv.org/abs/1503.03832).
Instead of using Viola and Jones' face detector, this repository uses a [implementation of the MTCNN model](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection) for both face detection and tracking initialization. It is also used for database creation.
(the original paper can be found [here](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf))

## Compatibility
The code has been tested using python 3.7.1 under 18.10 Ubuntu.

## Download
1. Clone this repo and cd to the 'models' directory.
2. Download either Facenet model from [Sandberg's repo](https://github.com/davidsandberg/facenet).
3. Run `python prepare_facenet.py path/to/facenet.zip` and cd to parent directory.

## Demo
1. Cd to the 'src' directory.
2. Run `python main.py demo/avengers.mp4 demo/user_data.csv demo/embeddings.npy`.

## Guide
The program consists of three parts: video file (webcam feed or downloaded video); user metadata and embeddings (both created from the *create_database.py* script). User metadata and embeddings derive from an image folder that contains all of the users to be identified. As a prerequisite you should have both video file and the images folder. Take into account that the image filename of a user will be used as his username.
1. Cd to the 'src' directory.
2. Run `python create_database.py output/user_data.csv output/embeddings.npy path/to/images/` to create embedddings.
3. Run `python main.py video_file.mp4 output/user_data.csv output/embeddings.npy` for face tracking and verification. If you wish to keep track of time a user appears in the video, run *screentime.py* instead of *main.py*.
