import cv2 as cv
import sys
import numpy as np

def imcv2_recolor(im, a=.1):

    t = np.random.uniform(-1, 1, 3)

    # random amplify each channel
    im = im.astype(np.float)
    im *= (1 + t * a)
    mx = 255. * (1 + a)
    up = np.random.uniform(-1, 1)
    im = np.power(im / mx, 1. + up * .5)

    return im

class object_detector:

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.framework = None
        self.load_model()

    def load_model(self):
        if self.model.endswith('weights') and self.cfg.endswith('cfg'):
            self.net = cv.dnn.readNetFromDarknet(self.cfg, self.model)
            self.framework = 'Darknet'
        elif self.model.endswith('caffemodel') and self.cfg.endswith('prototxt'):
            self.net = cv.dnn.readNetFromCaffe(self.cfg, self.model)
            self.framework = 'Caffe'
        else:
            sys.exit('Wrong input for model weights and cfg')

        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


    def predict(self,frame):

        # Create a 4D blob from a frame.
        if self.framework == 'Darknet':
            blob = cv.dnn.blobFromImage(cv.resize(frame, (416, 416)), 0.003921, (416, 416), (0,0,0), swapRB=True,  crop=False)
        else:
            blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

        # Run a model
        self.net.setInput(blob)
        out = self.net.forward()
        
        return out

    