

"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from Loading_Visualization.json_files_edit import *
from sort import convert_bbox_to_xyxy
import cv2
from math import isclose
import os
import numpy as np
from scipy.spatial import distance as dist
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)

def convert_bbox_to_xyxy_obj(bb):
    """"
    converts [y_top, y_bottom, x_left_ x_right] to the form [x1,y1,x2,y2]
    """


    bbn = [bb[2], bb[0], bb[3], bb[1]]

    return bbn

def get_digit(number,n):
    return str(number-int(number))[n]

def centroid_dist(bb_gt, bb_test):
    #inputCentroids = np.zeros((len(bb1), 2), dtype="float")
    #inputCentroids2 = np.zeros((len(bb2), 2), dtype="float")

    # use the bounding box coordinates to derive the centroid

    #bb_gt = np.expand_dims(bb_gt, 0)
    #bb_test = np.expand_dims(bb_test, 1)

    inputCentroids = np.zeros((len(bb_gt), 2))
    inputCentroids2 =np.zeros((len(bb_test), 2))

    for (i, (startX, startY, endX, endY)) in enumerate(bb_gt):
        # use the bounding box coordinates to derive the centroid
        cX = (startX + endX) / 2.0
        cY = (startY + endY) / 2.0
        inputCentroids[i] = (cX, cY)

    for (i, (startX, startY, endX, endY,_)) in enumerate(bb_test):
        # use the bounding box coordinates to derive the centroid
        cX = (startX + endX) / 2.0
        cY = (startY + endY) / 2.0
        inputCentroids2[i] = (cX, cY)


    D = dist.cdist(inputCentroids, inputCentroids2)


    return D


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.original_box=bbox

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.original_box=bbox

        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if (self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers,cost_fn, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """

    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)



    check = False

    if cost_fn is centroid_dist:
        check = True
        iou_matrix = centroid_dist(detections, trackers)
    else:
        iou_matrix = iou_batch(detections, trackers)


    if min(iou_matrix.shape) > 0:
        if check:
            a = (iou_matrix < iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(iou_matrix)
        else:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if check:
            if (iou_matrix[m[0], m[1]] > iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        else:
            if (iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort_obj(object):
    def __init__(self, max_age, min_hits, nb_frames, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.nb_frames = nb_frames
        self.trackers = []
        self.frame_count = 0

    def update(self, cost_fn, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, cost_fn,
                                                                                   self.iou_threshold)
        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i])
            self.trackers.append(trk)

        #print(f"len_trackers {len(self.trackers)}")
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.nb_frames):
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):
                self.trackers.pop(i)
        if (len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=10)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    parser.add_argument("--nb_frames", help="Nb of frames.", type=int, default=740)

    args = parser.parse_args()
    return args

def main_sort(detection_path,images_path, bboxes):
  # create instance of the SORT tracker
  ##min_hits more means new late registration and less zigzag track (min_hits = 6 means after registering new object
  ##they will check 6 frames if that detection is associated with previous ID or not,if not then new ID is
  ##registered after 6 frames later,late registration)
  for filename in os.listdir(detection_path):
      a=[]
      image_file = images_path + "image_" + filename[5:9] + ".jpg"  ##path of an image

      args = parse_args()
      mot_tracker =  Sort_obj(max_age=args.max_age,
                           min_hits=args.min_hits,
                          nb_frames=args.nb_frames,
                           iou_threshold=args.iou_threshold)

      # loop over the frames from the video stream
      det_path = detection_path + filename
      seq = np.array(bboxes[det_path][0])
      seq = seq[0:4]
      new_form = convert_bbox_to_xyxy((seq))
      a.append(new_form)
      img = cv2.imread(image_file)


      trackers = mot_tracker.update(iou_batch,a)
      print(f"new form {new_form}")
      print(f"trackers {trackers} + {filename}")
      for d in trackers:
          dh = 1200
          dw = 1920

          ID = d[4]

          x_left = int((d[0] * dw))
          y_top = int((d[1] * dh))
          x_right = int((d[2] * dw))
          y_bottom = int((d[3] * dh))

          # cv2.rectangle(img, (x_left_orig, y_top_orig), (x_right_orig, y_bottom_orig), (255, 0, 0), 2)

          cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (0, 0, 255), 2)
          text = "ID: {}".format(ID)
          cv2.putText(img, text, (x_left, y_top - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

      #cv2.imshow("Img", img)
      cv2.waitKey(0)


if __name__ == '__main__':
    cam_name = "s50_s_cam_near"

    folder = cam_name + "_detections"
    detection_path = 'C:/Users/cyrin/2021-06-29-13-49-20/{}/'.format(folder)

    folder2 = cam_name + "_images"
    images_path = 'C:/Users/cyrin/2021-06-29-13-49-20/{}/'.format(folder2)

    folder3 = cam_name + "_det_occ"
    occlusion_path = 'C:/Users/cyrin/2021-06-29-13-49-20/{}/'.format(folder3)

    bbox, position, classification_confidence, object_class = load_json_files_dic_of_all_bbox_of_all_images(detection_path)


    main_sort(detection_path,images_path, bbox)