import os

from drawing_package.drawing_package.data_interface.load_data import load_SensorOutput
from drawing_package.drawing_package.data_interface.save_data import save_SensorOutput
from occlusion_utils.centroid_tracker import CentroidTracker
import numpy as np


cams = ['s50_s_cam_far', 's50_s_cam_near', 's40_n_cam_near', 's40_n_cam_far']

#folder = '2021-12-03-09-10-52'
#dir = '../data/{}/'.format(folder)
folder = '2021-06-29-13-49-20'
dir = 'C:/Users/cyrin/{}/'.format(folder)

def assign_ids(data_list):

    tracker = CentroidTracker(10, 0.1)
    bbox = []

    for i in range(len(data_list)):
        bb_image=[]
        object_list = data_list[i].object_list
        for j in range(len(object_list)):
            bb_image.append(object_list[j].boundingbox)
        bbox.append(bb_image)

    for i in range(len(data_list)):
        bb = np.array(bbox[i])
        bb = bb[:, 0:4]
        objects, disap = tracker.update(bb)
        object_list = data_list[i].object_list

        for (objectId, bbx) in objects.items():
            if disap[objectId] == 0:
                for j in range(len(object_list)):
                    obj = object_list[j]
                    bb_obj = obj.boundingbox
                    bb_obj = np.array(bb_obj)
                    bb_obj = bb_obj[0:4]
                    if np.array_equal(bbx, bb_obj):
                        indice = objectId
                    else: continue

                    if hasattr(obj, 'label'):
                        data_list[i].object_list[j].label = indice
                    elif hasattr(obj, 'object_ID'):
                        data_list[i].object_list[j].object_ID = indice
                    elif hasattr(obj, 'object_id'):
                        data_list[i].object_list[j].object_id = indice

                    print(j)
    return data_list


if __name__ == '__main__':

    for cam in cams:
        print("--------------", cam)
        load_dir = dir + '{}_detections'.format(cam)
        data = load_SensorOutput(load_dir + "/")

        data['data_list'] = assign_ids(data['data_list'])

        save_dir = load_dir + "_ids"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        save_SensorOutput(data, save_dir + "/")


