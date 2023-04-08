from occlusion_utils.GTvsDT_undetected import *
import numpy as np
import os
from occlusion_utils.bbox_overlap_utils import *
from scripts.save_as_table import *



def area(bb):
    bb_area = abs((bb[3] - bb[2]) * (bb[1] - bb[0]))

    return bb_area

def combine_bbxs (a, b):

    x_left = min(a[0], b[0])
    y_top = min(a[0], b[0])
    x_right = max(a[3], b[3])
    y_bottom = max(a[1], b[1])

    comb_bb = [y_top, y_bottom, x_left,x_right]

    return comb_bb, area(comb_bb)

def bb_intersection_over_union(bb1, bb2):
    # determine the (x, y)-coordinates of the intersection rectangle

    x_left = max(bb1[2], bb2[2])
    y_top = max(bb1[0], bb2[0])
    x_right = min(bb1[3], bb2[3])
    y_bottom = min(bb1[1], bb2[1])

    intersection_area = abs(max((x_right - x_left, 0)) * max((y_bottom - y_top), 0))
    # intersection_area = (x_right - x_left) * (y_bottom - y_top)
    boxAArea = area(bb1)
    boxBArea = area(bb2)

    iou = intersection_area / float(boxAArea + boxBArea - intersection_area)

    # return the intersection over union value
    return iou

def check_2 (bb1,bb2):
    iou = bb_intersection_over_union(bb1,bb2)
    merge = False
    if iou>0.7:
        merge = True
    return merge


def check_m(match,occlus_param,stamp,frame_nb,match_id):

    lenOcc = len(occlus_param)
    ct_merge = 0
    ct_unmerged = 0
    a_merged = []
    a_unmerged = []
    ct_trunc= 0
    ct_untrunc = 0
    a_trun = []
    a_untrunc = []
    list_merged_trunc = []
    list_merged = []
    list_trunc = []
    list_check = []
    for box in match:
        list_check.append(box[1])
        list_check.append(box[0])

    for k in range(lenOcc):
        obj1 = occlus_param[k]
        lenA = len(obj1)
        for o in range(lenA):
            obj = obj1[o]
            obj3 = obj[8]
            obj2 = obj[9]
            if obj[0]:
                if 0.1<=obj[7]:
                    for box in match:
                        a_gt = area(box[0])
                        a_dt = area(box[1])
                        rank = match.index(box)
                        id_det = match_id[rank][1]
                        width_0 = box[0][3] - box[0][2]
                        height_0 = box[0][1] - box[0][0]
                        width_1 = box[1][3] - box[1][2]
                        height_1 = box[1][1] - box[1][0]
                        boxi = np.array(box[1])[0:4]

                        if a_dt < a_gt:
                            if (box[0] == obj3) or (box[0] == obj2):
                                if a_dt < (a_gt * 0.7)  and (box[0], box[1]) not in a_trun:
                                    ct_trunc += 1
                                    a_trun.append((box[0], box[1]))
                                    if (box[0] == obj3) :

                                        list_merged_trunc.append([obj[11],id_det, stamp, frame_nb, obj[7], obj[8],box[1], width_0,height_0,width_1,height_1, 'truncated'])
                                        list_trunc.append([obj[11], id_det, stamp, frame_nb, obj[7], obj[8], box[1], width_0,height_0,width_1,height_1, 'yes'])
                                    else:
                                        list_merged_trunc.append([obj[12],id_det, stamp, frame_nb, obj[7], obj[8],box[1], width_0,height_0,width_1,height_1, 'truncated'])
                                        list_trunc.append([obj[12], id_det, stamp, frame_nb, obj[7], obj[9], box[1],width_0,height_0,width_1,height_1, 'yes'])
                                else:
                                    ct_untrunc += 1
                                    a_untrunc.append((box[0], box[1]))
                                    if (box[0] == obj3):
                                        list_trunc.append( [obj[11], id_det, stamp, frame_nb, obj[7], obj[8], box[1], width_0,height_0,width_1,height_1, 'no'])
                                    else:  list_trunc.append( [obj[12], id_det, stamp, frame_nb, obj[7], obj[9], box[1], width_0,height_0,width_1,height_1, 'no'])
                        if width_1 > width_0 or height_1 > height_0:
                            if obj3 == box[0] :

                                #if  obj[10]:

                                comb, _ = combine_bbxs(box[0], obj[9])

                                if check_2(comb, boxi) and (box[0], box[1]) not in a_merged:
                                    ct_merge += 1
                                    a_merged.append((box[0], box[1]))
                                    list_merged_trunc.append( [obj[11], id_det, stamp, frame_nb, obj[7], obj[8], box[1], width_0,height_0,width_1,height_1, 'merged'])
                                    list_merged.append([obj[11], id_det, stamp, frame_nb, obj[7], obj[8], box[1],width_0,height_0,width_1,height_1, 'yes'])
                                else:
                                    ct_unmerged += 1
                                    a_unmerged.append((box[0], box[1]))
                                    list_merged.append([obj[11], id_det, stamp, frame_nb, obj[7], obj[8], box[1], width_0,height_0,width_1,height_1, 'no'])
                            elif obj2 == box[0] :
                                #if not obj[10]:

                                comb, _ = combine_bbxs(box[0], obj[8])
                                if check_2(comb, boxi) and (box[0], box[1]) not in a_merged:
                                    ct_merge += 1
                                    a_merged.append((box[0], box[1]))
                                    list_merged_trunc.append([obj[12], id_det, stamp, frame_nb, obj[7], obj[8], box[1],width_0,height_0,width_1,height_1, 'merged'])
                                    list_merged.append([obj[12], id_det, stamp, frame_nb, obj[7], obj[8], box[1], width_0,height_0,width_1,height_1, 'yes'])
                                else:
                                    ct_unmerged += 1
                                    a_unmerged.append((box[0], box[1]))
                                    list_merged.append([obj[12], id_det, stamp, frame_nb, obj[7], obj[8], box[1],width_0,height_0,width_1,height_1, 'no'])



    return ct_merge,ct_unmerged ,a_merged ,a_unmerged ,ct_trunc,ct_untrunc ,a_trun ,a_untrunc,list_merged_trunc,list_merged,list_trunc


def merged(bb_gt, det_path,gt_path,images_path,matched_pairs,stamps,ids):

    list_files = os.listdir(gt_path)
    len_f = len(list_files)
    det_occ, im = detection_overlap(bb_gt, images_path, gt_path, 1)
    occlusion = find_occlusion(bb_gt, det_occ, images_path, gt_path)
    ct_merged_images = 0
    ct_unmerged_images = 0
    ct_trunc_images = 0
    ct_untrunc_images = 0

    trunc_obj = []
    merged_obj = []
    list_merged = []
    list_trunc = []
    list_m_t = []


    for i in range(len_f):
        filename = gt_path + list_files[i]
        match = matched_pairs[filename]
        match_id = ids[filename]
        occlus_param = occlusion[i]
        stamp = stamps[filename]
        nb_frame = i

        ct_merge,ct_unmerged ,a_merged ,a_unmerged ,ct_trunc,ct_untrunc ,a_trun ,a_untrunc, list_merged_trunc,merged, trunc= check_m(match,occlus_param,stamp,nb_frame,match_id)
        list_merged.append(merged)
        list_trunc.append(trunc)
        list_m_t.append(list_merged_trunc)

        ct_merged_images += ct_merge
        ct_unmerged_images += ct_unmerged
        ct_trunc_images += ct_trunc
        ct_untrunc_images += ct_untrunc

        merged_obj.append(a_merged)
        trunc_obj.append(a_trun)

    return ct_merged_images, ct_unmerged_images, ct_trunc_images, ct_untrunc_images, merged_obj, trunc_obj,list_merged,list_trunc,list_m_t

def number_objec(data_list):
    counter = 0
    for i in range(len(data_list)):
        counter +=  data_list[i].num_detected

    return counter


def save_data(data_list,merged, trunc):

    for i in range(len(data_list)):

        list1_temp = []
        list2_temp = []

        merged_obj = merged[i]
        trunc_obj = trunc[i]

        check = data_list[i].object_list
        check1 =  data_list[i].object_list.copy()

        for box in merged_obj:
            list1_temp.append(box[0])

        for box in trunc_obj:
            list2_temp.append(box[0])

        len_occ1 = len(merged_obj)
        len_occ2 = len(trunc_obj)

        counter = data_list[i].num_detected
        for obj in check:
            if len_occ1== 0 and len_occ2==0 :
                check1.clear()
                counter = 0


            elif len_occ2 != 0 or len_occ1 != 0:

                if obj.boundingbox in list2_temp or obj.boundingbox in list1_temp :
                    continue
                else:
                    check1.remove(obj)

                    counter -= 1


        data_list[i].num_detected = counter
        data_list[i].object_list = check1

    return data_list


if __name__ == '__main__':


    folder5 = '2021-08-09-12-49-28'
    dir = 'C:/Users/cyrin/{}/'.format(folder5)


    cam_name = "s40_n_cam_far"

    folder = cam_name + "_detections_yolov7_ids_try"
    detection_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder)

    folder2 = cam_name + "_detections_yolov7_ids_new"
    gt_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder2)

    #folder = cam_name + "_detections_yolov7_modif_det"
    #detection_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder)


    folder7 = cam_name + "_images"
    image_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder7)

    bb_det,bb_gt, _, stamps = load(detection_path,gt_path)


    _, _, _, matched_pairs,id, _ = associate_GT_DT(bb_det, bb_gt,detection_path, gt_path)

    ct_merged_images, ct_unmerged_images, ct_trunc_images, ct_untrunc_images, merged_obj, trunc_obj, list_merged,list_trunc,list_m_t = merged(bb_gt, detection_path,gt_path,image_path,matched_pairs,stamps,id)
    #table(list_m_t, gt_path)

    #print(f"nb of undetected, {ct_undet}")

    #print(f"nb of undetected because of occ, {ct_unde_occ}")
    #print(f"nb of undetected not because of occ, {ct_unde_n_occ}")
    #print(f"nb of both undetected because of occ, {ct_undected_b_occ}")

    #plot_histo_sev(sev_nb,sev_nb1)

    load_dir = dir + '{}_detections_yolov7_ids_new'.format(cam_name)
    load_dir1 = dir + '{}_detections_yolov7_ids_try'.format(cam_name)
    load_dir2 = dir + '{}_detections_yolov7_ids_try_0'.format(cam_name)




    data = load_SensorOutput(load_dir + "/")
    data1 = load_SensorOutput(load_dir1 + "/")
    data2 = load_SensorOutput(load_dir2 + "/")


    counter = number_objec(data['data_list'])
    counter1 = number_objec(data1['data_list'])
    counter2 = number_objec(data2['data_list'])



    print(f"len_gt {len(data['data_list'])} | nb of objects_gt {cam_name} {counter}")
    print(f"nb of objects_det {cam_name} {counter1}")
    print(f"nb of objects_cor {cam_name} {counter2}")

    #data['data_list'] = save_data(data['data_list'],merged_obj,trunc_obj)

    #save_dir = load_dir + "_trunc_merge"
    #if not os.path.isdir(save_dir):
     #   os.mkdir(save_dir)
    #save_SensorOutput(data, save_dir + "/")



