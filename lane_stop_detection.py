import cv2
import numpy as np
import pickle
import time
from ultralytics import YOLO

FRAME_W, FRAME_H = 640, 480
CONF_THRESHOLD   = 0.40
LANE_HISTORY     = 10
MODEL_PATH       = "best.pt"
CAL_PATH         = "cal_pickle.p"
ROI_SRC          = np.float32([(0.42,0.63),(0.58,0.63),(0.14,0.87),(0.86,0.87)])
ROI_DST          = np.float32([(0,0),(1,0),(0,1),(1,1)])

try:
    cal     = pickle.load(open(CAL_PATH, "rb"))
    CAL_MTX, CAL_DIST = cal["mtx"], cal["dist"]
except FileNotFoundError:
    CAL_MTX = CAL_DIST = None

def undistort(img):
    return cv2.undistort(img, CAL_MTX, CAL_DIST, None, CAL_MTX) if CAL_MTX is not None else img

_src = ROI_SRC * np.float32([FRAME_W, FRAME_H])
_dst = ROI_DST * np.float32([FRAME_W, FRAME_H])
M    = cv2.getPerspectiveTransform(_src, _dst)
Minv = cv2.getPerspectiveTransform(_dst, _src)

def to_birdseye(img): return cv2.warpPerspective(img, M,    (FRAME_W, FRAME_H))
def to_camera(img):   return cv2.warpPerspective(img, Minv, (FRAME_W, FRAME_H))

def threshold(img):
    hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white  = cv2.inRange(hsv, (0,  0,  180), (255, 40, 255))
    yellow = cv2.inRange(hsv, (18, 80, 100), (35, 255, 255))
    return cv2.bitwise_or(white, yellow)

_la, _lb, _lc = [], [], []
_ra, _rb, _rc = [], [], []

def fit_lanes(bird):
    global _la, _lb, _lc, _ra, _rb, _rc

    hist = np.sum(bird[bird.shape[0]//2:], axis=0)
    mid  = hist.shape[0] // 2
    lx   = np.argmax(hist[:mid])
    rx   = np.argmax(hist[mid:]) + mid

    nwin, margin, minpix = 10, 60, 15
    win_h = bird.shape[0] // nwin
    nz    = bird.nonzero()
    nzy, nzx = np.array(nz[0]), np.array(nz[1])
    l_inds, r_inds = [], []

    for w in range(nwin):
        y0 = bird.shape[0] - (w+1)*win_h
        y1 = bird.shape[0] -  w   *win_h
        gl = ((nzy>=y0)&(nzy<y1)&(nzx>=lx-margin)&(nzx<lx+margin)).nonzero()[0]
        gr = ((nzy>=y0)&(nzy<y1)&(nzx>=rx-margin)&(nzx<rx+margin)).nonzero()[0]
        l_inds.append(gl); r_inds.append(gr)
        if len(gl) > minpix: lx = int(np.mean(nzx[gl]))
        if len(gr) > minpix: rx = int(np.mean(nzx[gr]))

    lxp = nzx[np.concatenate(l_inds)]; lyp = nzy[np.concatenate(l_inds)]
    rxp = nzx[np.concatenate(r_inds)]; ryp = nzy[np.concatenate(r_inds)]

    if not (lxp.size and rxp.size):
        return None, None, None

    lf = np.polyfit(lyp, lxp, 2); rf = np.polyfit(ryp, rxp, 2)
    _la.append(lf[0]); _lb.append(lf[1]); _lc.append(lf[2])
    _ra.append(rf[0]); _rb.append(rf[1]); _rc.append(rf[2])
    _la,_lb,_lc = _la[-LANE_HISTORY:],_lb[-LANE_HISTORY:],_lc[-LANE_HISTORY:]
    _ra,_rb,_rc = _ra[-LANE_HISTORY:],_rb[-LANE_HISTORY:],_rc[-LANE_HISTORY:]

    ploty  = np.linspace(0, bird.shape[0]-1, bird.shape[0])
    left_x = np.mean(_la)*ploty**2 + np.mean(_lb)*ploty + np.mean(_lc)
    right_x= np.mean(_ra)*ploty**2 + np.mean(_rb)*ploty + np.mean(_rc)
    offset = (bird.shape[1]/2) - (left_x[-1]+right_x[-1])/2   # + = drift right

    return left_x, right_x, offset

def draw(img, left_x, right_x, offset, fps):
    ploty   = np.linspace(0, FRAME_H-1, FRAME_H)
    overlay = np.zeros_like(img)
    pts     = np.hstack([
        np.array([np.transpose(np.vstack([left_x,  ploty]))]),
        np.array([np.flipud(np.transpose(np.vstack([right_x, ploty])))]),
    ])
    cv2.fillPoly(overlay, np.int_(pts), (0, 200, 255))
    img = cv2.addWeighted(img, 0.8, to_camera(overlay), 0.4, 0)

    # minimal HUD
    label = "LEFT" if offset < -20 else "RIGHT" if offset > 20 else "STRAIGHT"
    cv2.putText(img, f"{label}  {offset:+.0f}px", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(img, f"FPS {fps:.0f}", (FRAME_W-90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    return img

model = YOLO(MODEL_PATH)
cap   = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

t0, fid = time.time(), 0

while True:
    ret, frame = cap.read()
    if not ret: break
    fid += 1

    img  = cv2.resize(undistort(frame), (FRAME_W, FRAME_H))
    bird = to_birdseye(threshold(img))

    left_x, right_x, offset = fit_lanes(bird)

    if left_x is not None:
        img = draw(img, left_x, right_x, offset, fid/(time.time()-t0))
    else:
        cv2.putText(img, "NO LANE", (240, FRAME_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    detections = model(img, conf=CONF_THRESHOLD, verbose=False)
    cv2.imshow("car", detections[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()