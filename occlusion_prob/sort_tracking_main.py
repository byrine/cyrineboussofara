from occlusion_utils.sort import *
import matplotlib
matplotlib.use('TkAgg')

import cv2
import os
from Loading_Visualization.json_files_edit import *
from Loading_Visualization.draw_occ import *


cam_name = "s40_n_cam_far"

folder = cam_name + "_detections"
detection_path = 'C:/Users/cyrin/2021-06-29-13-49-20/{}/'.format(folder)

folder2= cam_name + "_images"
images_path = 'C:/Users/cyrin/2021-06-29-13-49-20/{}/'.format(folder2)

folder3= cam_name + "_det_occ"
occlusion_path = 'C:/Users/cyrin/2021-06-29-13-49-20/{}/'.format(folder3)


def main():

    bbox, position, classification_confidence, object_class = load_json_files_dic_of_all_bbox_of_all_images(detection_path)
    print("loading done")


    #sort

    bb_pred_sort, bbox1= main_sort(detection_path, images_path, bbox)
    #print(bb_pred_sort)
    print("assignment of ids & prediction of bbox done")
    det_occ_sort,im = detection_overlap(bbox1, images_path, detection_path, 1)
    det_occ_pred_sort, im = detection_overlap(bb_pred_sort, images_path, detection_path,3)
    #print(det_occ_pred_sort[4])
    print("prediction of occlusion for sequence done")
    occlusion_sort = find_occlusion(bbox1, det_occ_sort, images_path, detection_path)
    print(occlusion_sort[1])

    #print(det_occ_sort[0])
    l = len(occlusion_sort)
    print("finding occlusions done")
    gt_sort = gt_ids_occluded(occlusion_sort, detection_path,im,l)
    list = predict_occlusion_one_frame(bbox, bbox1, detection_path,im,0.002)
    print(list[0])
    print("prediction of occlusion for frame done")
    frame_sort = conv_list_to_dic(detection_path,list,im)
    seq_sort = conv_list_to_dic(detection_path,det_occ_pred_sort,im)
    #plot_curve_comp(detection_path,frame_sort,seq_sort,gt_sort,im)


    if not os.path.exists(occlusion_path):

        try:
            os.mkdir(occlusion_path)

        except OSError:
            print("Creation of the directory %s failed" % occlusion_path)
        else:
            print("Successfully created the directory %s " % occlusion_path)

    #create_json_files(cam_name, id_obj,bbox,bb_pred, occlusion, occlusion_pred, detection_path, occlusion_path, position, classification_confidence, object_class,im )
    print("Stored everything")


if __name__ == '__main__':
    main()