#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import numpy as np
import cv2
import video_23 as video
from common_23 import anorm2, draw_str
from time import clock

lk_params = dict(winSize=(10, 10), #play around with these
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


feature_params = dict(maxCorners=50,
                      qualityLevel=0.6,
                      minDistance=7,
                      blockSize=15)


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        print(video_src)
        self.frame_idx = 0

    def run(self):
        framec = 0
        ret, frame = self.cam.read()      
                          
        while True:            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            print(len(self.tracks))

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2) #points to track
                
                # Insert your Optical Flow algorithms here
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    img0, img1, p0, None, **lk_params) #flow between img0 and img1 with p0 as feature points
                p0r, st, err = cv2.calcOpticalFlowPyrLK(
                    img1, img0, p1, None, **lk_params) #flow between img1 and img0 with p1 as feature points
                
                print(p0.shape)
                d = abs(p0-p0r).reshape(-1, 2).max(-1) #threshold to find good points to track

                good = d < 1 #keep good points

                print("----")
                print(good)

                #  to draw the tracks
                new_tracks = [] 
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue

                    tr.append((x, y))
                    
                    if len(tr) > self.track_len:
                        del tr[0]
                    
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 4, (0, 255, 0), -1)

                self.tracks = new_tracks #new points to track from the next frame onwards
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0), 3)
                draw_str(vis, 20, 20, 'track count: %d' % len(self.tracks))

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(
                    frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('Sparse Lucas Kanade', vis)
            fn = "out/lkfilem_absurd"+str(framec).rjust(4, '0')+".png"
            cv2.imwrite(fn, vis)
            framec = framec+1

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
            else:
                ret, frame = self.cam.read()
                if frame is None:                    
                    break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
