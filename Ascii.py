import cv2
import numpy as np

def rescaleFrame(frame, dim = 50):
    dimension = (dim, dim)
    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

def getValue(frame,size=15):
    s = 'N@#W$9876543210?!abc;:+=-,._ '[::-1]
    scale = len(s)
    black = np.zeros((frame.shape[1]*size,frame.shape[0]*size,3),dtype='uint8')
    for i in range(frame.shape[0]):
        for j in range(frame.shape[0]):
            value = (scale*frame[i][j])//255
            if value == scale:
                value=scale-1
            black = cv2.putText(black,str(s[value]),(j*size+2,i*size+10),cv2.FONT_HERSHEY_COMPLEX,0.4,(98,160,3))
    return black


def ascii(frame):
    frame = cv2.flip(frame, 1)
    rescaled = rescaleFrame(frame)
    grey = cv2.cvtColor(rescaled,cv2.COLOR_BGR2GRAY)
    valued = getValue(grey)
    cv2.imshow("img",valued)
    cv2.imshow("rescaled",frame)

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    _, frame = cap.read()
    ascii(frame)
    if cv2.waitKey(20) == ord('q'):
        break