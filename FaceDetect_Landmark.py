# !/usr/bin/python

# Copyright (c) 2015 Matthew Earl
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
#
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This is the code behind the Switching Eds blog post:
    http://matthewearl.github.io/2015/07/28/switching-eds-with-python/
See the above for an explanation of the code below.
To run the script you'll need to install dlib (http://dlib.net) including its
Python bindings, and OpenCV. You'll also need to obtain the trained model from
sourceforge:
    http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2
Unzip with `bunzip2` and change `PREDICTOR_PATH` to refer to this file. The
script is run like so:
    ./faceswap.py <head image> <face image>
If successful, a file `output.jpg` will be produced with the facial features
from `<head image>` replaced with the facial features from `<face image>`.
"""

import cv2
import dlib

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 68))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im):
    rect = detector(im, 1)

    points = []

    if len(rect) > 1:      # raise TooManyFaces
        return

    if len(rect) == 0:     # raise NoFaces
        return

    for p in predictor(im, rect[0]).parts():
        points.append((p.x, p.y))

    return points


def read_im_and_landmarks(im):

    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR, im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)

    return s


if __name__ == '__main__':

    color_point = (255, 0, 0)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, im2 = cap.read()
        landmarks2 = read_im_and_landmarks(im2)

        if landmarks2 is not None:
            # for i in range(landmarks2.__len__()):
            for i in MOUTH_POINTS:
                (a, b) = landmarks2[i]
                cv2.circle(im2, (a, b), 3, color_point, -1)

        cv2.imshow("Face Swapped", im2)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
