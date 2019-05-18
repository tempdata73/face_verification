# Real time face tracking and verification
This is an Opencv-tensorflow-mxnet implementation of face tracking and verification on real time using either CPU or GPU.
The pretrained models used for face verification were obtained from [David Sandberg's github repo](https://github.com/davidsandberg/facenet)
which are tensorflow implementations of [FaceNet](https://arxiv.org/abs/1503.03832).
Instead of using Viola and Jones' face detector, this repository uses a builtin Opencv DNN model to initialize 
the trackers and a [implementation of the MTCNN model](https://github.com/YYuanAnyVision/mxnet_mtcnn_face_detection)
(the original paper can be found [here](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)) to create the
user database.

## Compatibility
The code has been tested using python 3.7.1 under 18.10 Ubuntu.

## Guide
1. Clone this repository and change to the "models" directory.
2. Download either one of the pretrained models from [Sandberg's repo](https://github.com/davidsandberg/facenet).
3. Run ```
       python prepare_facenet.py path/to/model.zip
       ``` on the terminal.
4. Cd to the parent directory.
5. Create an "images" directory and add images to it (it is recommendable to add an image of you for the first test).
   The filename of each image will be the username of the person that appears in it.
6. Run ```
       python create_database.py user_data.csv embeddings.npy path/to/images/
       ```. The csv file contains the names of each user and the path to his respective image, the npy file contains
       the embeddings created for each image.
7. Run ```
       python main.py user_data.csv embeddings.npy -t tracker -n interval
       ``` to do real time face tracking. If you're on the database, then the tracker should write your name
       on top of the bounding box, otherwise it'll write "unassigned." The tracker flag refers to which tracker
       to use (one of all available OpenCV trackers) and the interval flag refers to the frequency on which to
       do detection to check if new faces have appeared/disappeared. Defaults are kcf and 20, respectively.
