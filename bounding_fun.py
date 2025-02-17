import cv2
import numpy as np
from skimage.morphology import square, closing

def airlight(I, size=3):
    return np.max(I, axis=(0, 1)) / 255

def cal_transmission2(I, est_tr_proposed, r, omega):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    transmission = cv2.morphologyEx(est_tr_proposed, cv2.MORPH_CLOSE, kernel)
    return transmission

def defog(I, transmission, A, t_min=0.95):
    J = (I - A) / np.maximum(transmission, t_min)[:, :, np.newaxis] + A
    return np.clip(J, 0, 1)

def bounding_function(I, zeta):
    I = I.astype(np.float64) / 255.0
    min_I = np.min(I, axis=2)
    MAX = np.max(min_I)
    
    A1 = airlight(I, 3)
    A = np.max(A1)
    
    delta = zeta / (min_I ** 0.5)
    est_tr_proposed = 1 / (1 + (MAX * 10 ** (-0.05 * delta)) / (A - min_I))
    
    tr1 = min_I >= A
    tr2 = min_I < A
    tr2 = np.abs(tr2 * est_tr_proposed)
    tr4 = np.abs(est_tr_proposed * tr1)
    
    tr3_max = np.max(tr4)
    if tr3_max == 0:
        tr3_max = 1
    
    tr3 = tr4 / tr3_max
    est_tr_proposed = tr2 + tr3
    
    est_tr_proposed = closing(est_tr_proposed, square(3))
    est_tr_proposed = cal_transmission2(I, est_tr_proposed, 1, 0.5)
    
    r = defog(I, est_tr_proposed, A1, 0.95)
    
    return r, est_tr_proposed, A
