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

import cv2
import dlib

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

ALL_POINTS = range(0, 68)
FACE_POINTS = range(17, 68)
MOUTH_POINTS = range(48, 60)
LIP_POINTS = range(60, 68)
RIGHT_BROW_POINTS = range(17, 22)
LEFT_BROW_POINTS = range(22, 27)
RIGHT_EYE_POINTS = range(36, 42)
LEFT_EYE_POINTS = range(42, 48)
NOSE_POINTS = range(27, 35)
JAW_POINTS = range(0, 17)

color_red = (0, 0, 255)
color_blue = (255, 0, 0)
color_white = (255, 255, 255)
color_black = (0, 0, 0)

yawning_limit = 0.35

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


def get_landmarks(im):

    rect = detector(im, 1)

    if len(rect) != 1:      # raise TooManyFaces or NoFaces
        return

    points = []
    predict_ret = predictor(im, rect[0]).parts()
    for p in predict_ret:
        points.append((p.x, p.y))

    return points


if __name__ == '__main__':

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, im2 = cap.read()

        # ---------- get the camera video size ---------------
        height, width = im2.shape[:2]

        landmarks2 = get_landmarks(im2)
        cv2.rectangle(im2, (30, height - 30), (200, height - 30), color_black, 54)
        cv2.rectangle(im2, (30, height - 30), (200, height - 30), color_white, 50)

        if landmarks2 is None:
            alarm_str = "No Detect!"
        else:
            # ---------- draw the landmarks data ------------
            for i in LIP_POINTS:
                cv2.circle(im2, landmarks2[i], 3, color_blue, -1)

            # ------------- decide the yawning --------------
            rate_mouth = float(landmarks2[66][1] - landmarks2[62][1]) / (landmarks2[64][0] - landmarks2[60][0])
            if rate_mouth > yawning_limit:
                alarm_str = "Yawning!"
            else:
                alarm_str = "Ok!"

        # ------------ display the alarm string --------------
        cv2.putText(im2, alarm_str, (20, height - 20), cv2.FONT_HERSHEY_DUPLEX, 1, color_red, 2)
        cv2.imshow("Face Swapped", im2)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
