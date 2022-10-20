from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np

FRAME_SIZE = 480
FPS = 60.0
FRAME_LENGTH = 120
POLE = 2
ROTOR_R = FRAME_SIZE//3
SHAFT_R = ROTOR_R//5*4

VECTOR_R = FRAME_SIZE//5*2


def RotorImg(im, q):
    im = np.ones((FRAME_SIZE,FRAME_SIZE, 3), dtype=np.uint8)*255 if im is None else im
    p0 = FRAME_SIZE//2
    for i in range(POLE//2):
        cv2.ellipse(im, (p0,p0), (ROTOR_R, ROTOR_R), 0, q - 360/POLE/2 + (2*i)*360/POLE, q + 360/POLE/2 + (2*i)*360/POLE, (0,0,255), thickness=-1)
        cv2.ellipse(im, (p0,p0), (ROTOR_R, ROTOR_R), 0, q - 360/POLE/2 + (2*i+1)*360/POLE, q + 360/POLE/2 + (2*i+1)*360/POLE, (255,0,0), thickness=-1)
        cv2.circle(im, (p0,p0), SHAFT_R, (127,127,127), thickness=-1)
    return im

def VectorImg(im, v, d=1, color=(0,0,255), label=""):
    im:np.ndarray = np.ones((FRAME_SIZE,FRAME_SIZE, 3), dtype=np.uint8)*255 if im is None else im
    p0 = FRAME_SIZE//2
    r = FRAME_SIZE//25
    cv2.arrowedLine(im, (p0,p0), (p0+int(v[0]), p0+int(v[1])), color, thickness=d)
    cv2.putText(im, label, (p0+int(v[0]+r*np.cos(np.arctan2(v[1], v[0]))), p0+int(v[1]+r*np.sin(np.arctan2(v[1], v[0])))), cv2.FONT_HERSHEY_COMPLEX, 1.0, color)
    return im

def overlay(im_base, im_overlay, pos=(0,0)):
    im_gray = cv2.cvtColor(im_overlay,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(im_gray, 1, 255, cv2.THRESH_BINARY)
    im_base[pos[0]:mask.shape[0], pos[1]:mask.shape[1], :] = cv2.bitwise_and(im_base[pos[0]:mask.shape[0], pos[1]:mask.shape[1],:], im_base[pos[0]:mask.shape[0], pos[1]:mask.shape[1],:], mask=cv2.bitwise_not(mask))
    im_base[pos[0]:mask.shape[0], pos[1]:mask.shape[1],:] += im_overlay
    return im_base



fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
video  = cv2.VideoWriter('RotorImage.mp4', fourcc, FPS, (FRAME_SIZE, FRAME_SIZE))

ax = VectorImg(np.zeros((FRAME_SIZE,FRAME_SIZE, 3), dtype=np.uint8), (VECTOR_R, 0),d=2, color=(255,255,0), label="d")
ax = VectorImg(ax, (VECTOR_R*np.cos(np.pi/POLE), -VECTOR_R*np.sin(np.pi/POLE)),d=2, color=(255,0,255), label="q")
for i in range(FRAME_LENGTH):
    A = cv2.getRotationMatrix2D((ax.shape[0]//2, ax.shape[1]//2), i*3, 1.0)
    ax_warp = cv2.warpAffine(ax, A, ax.shape[0:2])
    img = RotorImg(None, -i*3)
    img = overlay(img, ax_warp)
    video.write(img)


video.release()