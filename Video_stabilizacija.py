# 26. Stabilizacija videa
# Razviti algoritam koji će korištenjem optičkog toka, point feature matching 
# ili drugih proizvoljnih značajki/metoda ispraviti micanje kamere tijekom 
# snimanja videa. Opisati korištenu metodu, njene prednosti i nedostatke i
# ispitati na vlastitim primjerima. Nije dozvoljeno koristiti gotove 
# biblioteke za stabilizaciju videa, ali dozvoljeno 
# je koristiti OpenCV metode ili sl. biblioteke za pronalazak optičkog toka, 
# značajki i sl.

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

SMOOTHING_RADIUS=50 

def movingAverage(curve, radius): 
  window_size = 2 * radius + 1
  f = np.ones(window_size)/window_size 

  curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 

  curve_smoothed = np.convolve(curve_pad, f, mode='same') 
  curve_smoothed = curve_smoothed[radius:-radius]

  return curve_smoothed 

def smooth(trajectory): 
  smoothed_trajectory = np.copy(trajectory) 

  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)

  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape

  T = cv.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv.warpAffine(frame, T, (s[1], s[0]))
  return frame


cap = cv.VideoCapture('video.mp4')

n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv.CAP_PROP_FPS))

w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*'MP4V')

out = cv.VideoWriter('video_out.mp4', fourcc, fps, (w, h))

_, prev = cap.read()

prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
plt.imshow(prev_gray, cmap="gray")
plt.show()

transforms = np.zeros((n_frames-1, 3), np.float32)

# Good features to track example
prev_pts1 = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
prev_pts1 = np.int0(prev_pts1)

for i in prev_pts1:
    x, y = i.ravel()
    cv.circle(prev_gray, (x, y), 3, 255, -1)

plt.imshow(prev_gray, cmap="gray")
plt.show()
#

for i in range(n_frames-2):
    prev_pts = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    succes, curr = cap.read()
    if not succes:
        break

    curr_gray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
        
    curr_pts, status, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
    

    assert prev_pts.shape == curr_pts.shape

    idx = np.where(status == 1)[0]
    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    m = cv.estimateAffine2D(prev_pts, curr_pts)[0]

    dx = m[0,2]
    dy = m[1,2]

    da = np.arctan2(m[1,0], m[0,0])

    transforms[i] = [dx, dy, da]

    prev_gray = curr_gray
    
    print("Frame: " + str(i) + "/" + str(n_frames) + " - Tracked points : " + str(len(prev_pts)))

trajectory = np.cumsum(transforms, axis=0)
plt.plot(trajectory)
plt.show()

smoothed_trajectory = smooth(trajectory)
plt.plot(smoothed_trajectory)
plt.show()

difference = smoothed_trajectory - trajectory
transforms_smooth = transforms + difference


cap.set(cv.CAP_PROP_POS_FRAMES, 0)
    
for i in range(n_frames-2):
    succes, frame = cap.read()
    if not succes:
        break
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    m = np.zeros((2,3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy

    frame_stabilized = cv.warpAffine(frame, m, (w,h))

    frame_stabilized = fixBorder(frame_stabilized)

    frame_out = cv.hconcat([frame, frame_stabilized])

    if(frame_out.shape[1] > 1920):
        frame_out = cv.resize(frame_out,  (int(frame_out.shape[1]/2), int(frame_out.shape[0]/2)))
    cv.imshow("Before and After", frame_out)
    cv.waitKey(10)
    out.write(frame_stabilized)

cap.release()
out.release()

cv.destroyAllWindows()






