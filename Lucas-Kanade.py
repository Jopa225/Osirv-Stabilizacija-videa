import numpy as np
import cv2 as cv

cap = cv.VideoCapture('video.mp4')

_, prev = cap.read()

prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

color = np.random.randint(0, 255, (200, 3))
mask = np.zeros_like(prev)

feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


prev_pts = cv.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

while(1):
    succes, curr = cap.read()
    if not succes:
        break

    curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)

    # Lucas-Kanade optical flow
    curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
    
    if curr_pts is not None:
        good_new = curr_pts[status==1]
        good_old = prev_pts[status==1]
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        curr = cv.circle(curr, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(curr, mask)
    cv.imshow('frame', img)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    prev_gray = curr_gray.copy()
    prev_pts = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()