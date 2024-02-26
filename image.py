import cv2
from settings import *


def prepare():
    cap = cv2.VideoCapture(0)
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15.,
                          (640, 480)) if SAVE_VIDEO else None
    return cap, out



def draw_boxes(frame, boxes):
    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                      (255, 0, 0), 2)


def display_image(frame):
    cv2.imshow('press Esc to exit', frame)


def close(cap, out):
    cap.release
    if SAVE_VIDEO:
        out.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)