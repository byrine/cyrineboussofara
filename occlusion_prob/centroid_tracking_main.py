from occlusion_utils.bbox_overlap_utils import *
import matplotlib
matplotlib.use('TkAgg')

import cv2
import os
from Loading_Visualization.json_files_edit import load_json_files_with_ids as load
from Loading_Visualization.json_files_edit import load_json_files_dic_of_all_bbox_of_all_images

from Loading_Visualization.draw_occ import *


cam_name = "s40_n_cam_far"

folder = cam_name + "_detections_yolov7_ids_try"
detection_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder)

folder2 = cam_name + "_detections_yolov7_ids_new"
gt_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder2)

#folder = cam_name + "_detections_yolov7_modif_det"
#detection_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder)

folder3= cam_name + "_det_occ"
occlusion_path = 'C:/Users/cyrin/2021-06-29-13-49-20/{}/'.format(folder3)

folder7 = cam_name + "_images"
image_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder7)


def main():

    bbox, position, classification_confidence, object_class = load_json_files_dic_of_all_bbox_of_all_images(detection_path)

    bb_det, bb_gt, _, stamps1 = load(detection_path, gt_path)

    print("loading done")

    #centroid tracking + prediction

    id_obj= object_id_cons(bbox,image_path, detection_path)
    #print(id_obj)
    print("assignment of ids done")
    bb_pred = predict_occlusion_sequence(bb_det,bbox,image_path, detection_path)
    #print(bb_pred)
    print("prediction of bboxes done")
    det_occ,im = detection_overlap(bb_gt, image_path, detection_path,1)
    #print(f"det {det_occ[1]}")
    det_occ_pred,im = detection_overlap(bb_pred, image_path, detection_path,4)
    #print(det_occ_pred[16])

    print("prediction of occlusion for sequence done")
    #find occlusion in each image from detected bboxes
    occlusion = find_occlusion(bb_gt,det_occ,image_path, detection_path)
    print("finding occlusions done")
    l = len(occlusion)
    #print(occlusion[17])

    gt = gt_ids_occluded(occlusion, detection_path,im,l)
    #prediction of occlusion with one frame
    list= predict_occlusion_one_frame(bbox,bb_det, detection_path,im,0.2)
    print("prediction of occlusion for 1 frame done")
    #print(list[16])
    frame1 = conv_list_to_dic(detection_path,list,im)
    #print(frame1)
    seq = conv_list_to_dic(detection_path,det_occ_pred,im)
    plot_curve_comp(detection_path,frame1,seq,gt,im)
    #occlusion_pred = find_occlusion(bb_pred,det_occ_pred,images_path, detection_path)



    if not os.path.exists(occlusion_path):

        try:
            os.mkdir(occlusion_path)

        except OSError:
            print("Creation of the directory %s failed" % occlusion_path)
        else:
            print("Successfully created the directory %s " % occlusion_path)

    #create_json_files(cam_name, id_obj,bbox,bb_pred, occlusion, occlusion_pred, detection_path, occlusion_path, position, classification_confidence, object_class,im  )
    print("Stored everything")




    #plot_annotated_grid(image_path, occlusion_path)







if __name__ == '__main__':
    main()



