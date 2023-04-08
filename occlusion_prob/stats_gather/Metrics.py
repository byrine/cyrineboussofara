import numpy as np

from Loading_Visualization.json_files_edit import load_json_files_with_ids as load
from occlusion_utils.GTvsDT_undetected import *
from occlusion_utils.GTvsDT_merged_trun import *

def iou_GT(merged_obj, trunc_obj):
    iou_merged = []
    iou_trunc = []

    for i in range(len(merged_obj)):
        iou = []
        iou2 = []
        merge = merged_obj[i]
        trun = trunc_obj[i]

        if len(merged_obj[i]) !=0:
            for j in range(len(merge)):
                mer = merge [j]
                joli = bb_intersection_over_union(mer[0],mer[1])
                iou.append((joli,mer[0],mer[1]))
        if len(trunc_obj[i]) !=0:
            for j in range(len(trun)):
                tr = trun [j]
                joli1 = bb_intersection_over_union(tr[0],tr[1])
                iou2.append((joli1,tr[0],tr[1]))

        iou_merged.append(iou)
        iou_trunc.append(iou2)
    return iou_merged,iou_trunc

def iou_GT1(match_pairs,gt_path):

    iou_tot =[]

    lala = os.listdir(gt_path)
    len_f = len(lala)
    for i in range(len_f):
        iou = []
        filename =  gt_path + lala[i]
        matcho = match_pairs[filename]
        for mer in matcho:

            joli = bb_intersection_over_union(mer[0],mer[1])
            iou.append((joli,mer[0],mer[1]))
        iou_tot.append(iou)




    return iou_tot

def trans_in_list(list_a_remplir, iou_a_remplir, liste) :

    for box in liste:
        #boxi = np.array(box[2])[0:4]
        #boxi = boxi.tolist()
        list_a_remplir.append(box[2][0:4])
        iou_a_remplir.append(box[0])
    return list_a_remplir, iou_a_remplir


def delta_GT(iou_merged,iou_trunc, iou_merged1,iou_trunc1,iou_tot):

    delta_iou_merge = []
    delta_iou_trunc = []
    ct_objs_false_from_beg = 0
    ct_objs_merg_from_beg = 0
    ct_objs_false_from_begin = 0
    ct_objs_merg_from_begin = 0
    for l in range(len(iou_merged)):
        iou = []
        iou2 = []
        list_box_merge_not_corr = []
        list_box_merge_corr = []
        iou_corr_merge = []
        iou_corr_not_merge = []
        list_box_trunc_not_corr = []
        list_box_trunc_corr = []
        iou_corr_trunc = []
        iou_corr_not_trunc = []
        list_box_corr = []
        iou_corr = []
        iou_par = iou_tot[l]
        merge = iou_merged[l]
        trun = iou_trunc[l]
        merge1 = iou_merged1[l]
        trun1 = iou_trunc1[l]
        #delta_iou_merge = []
        #delta_iou_trunc = []
        #ct_objs_false_from_beg = 0
        #ct_objs_merg_from_beg = 0

        delta_iou_mergein = []
        delta_iou_truncin = []
        #ct_objs_false_from_begin = 0
        #ct_objs_merg_from_begin = 0
        list_box_merge_not_corr, iou_corr_not_merge =  trans_in_list(list_box_merge_not_corr, iou_corr_not_merge,merge )
        list_box_merge_corr, iou_corr_merge = trans_in_list(list_box_merge_corr, iou_corr_merge, merge1)

        list_box_corr, iou_corr = trans_in_list(list_box_corr, iou_corr, iou_par)

        list_box_trunc_not_corr, iou_corr_not_trunc = trans_in_list(list_box_trunc_not_corr, iou_corr_not_trunc, trun)
        list_box_trunc_corr, iou_corr_trunc = trans_in_list(list_box_trunc_corr, iou_corr_trunc, trun1)


        if len(list_box_merge_not_corr) != 0 and len(list_box_corr) !=0:
            for j in range(len(list_box_merge_not_corr)):
                #if list_box_merge_not_corr[j] in list_box_merge_corr:
                #for i in range(len(list_box_merge_corr)):
                    #diff = np.array(list_box_merge_not_corr[j])- np.array(list_box_merge_corr[i])
                for i in range(len(list_box_corr)):
                    if np.isclose(list_box_merge_not_corr[j],list_box_corr[i], rtol=0, atol=0.0005).any():
                        delta_iou = iou_corr[i] - iou_corr_not_merge[j]
                        if abs(delta_iou) not in delta_iou_mergein:
                            delta_iou_mergein.append((delta_iou))

        if len(list_box_trunc_not_corr) != 0 and len(list_box_trunc_corr)!=0:
            for j in range(len(list_box_trunc_not_corr)):
                #if list_box_trunc_not_corr[j] in list_box_trunc_corr:
                #for i in range(len(list_box_trunc_corr)):
                    #diff = np.array(list_box_trunc_not_corr[j]) - np.array(list_box_trunc_corr[i])
                for i in range(len(list_box_corr)):

                    if np.isclose(list_box_trunc_not_corr[j],list_box_corr[i], rtol=0, atol=0.0009).any():
                        delta_iou = iou_corr[i] - iou_corr_not_trunc[j]

                        delta_iou_truncin.append((delta_iou))


        if len(list_box_trunc_not_corr) != 0 and len(list_box_trunc_corr)!=0:
            for j in range(len(list_box_trunc_not_corr)):
                if list_box_trunc_not_corr[j] in list_box_trunc_corr:
                    ct_objs_false_from_begin +=1
                for n in range(len(list_box_corr)):
                   if (n+1)<len(list_box_corr) and np.isclose(list_box_trunc_not_corr[j], list_box_corr[n+1], rtol=0, atol=0.0009).any():
                       ct_objs_false_from_begin += 1



        if len(list_box_merge_not_corr) != 0 and len(list_box_merge_corr) !=0:
            for j in range(len(list_box_merge_not_corr)):
                if list_box_merge_not_corr[j] in list_box_merge_corr:
                    ct_objs_merg_from_begin +=1



        delta_iou_merge.append(delta_iou_mergein)
        delta_iou_trunc.append(delta_iou_truncin)


    return delta_iou_merge, delta_iou_trunc,ct_objs_false_from_begin, ct_objs_merg_from_begin


def gather_ious(delta_iou):

    count_ps = 0
    count_ns = 0
    count_unchanged = 0

    sev = []

    for i in range(len(delta_iou)):
        if delta_iou[i]:
            no = delta_iou[i]
            for r in range(len(no)):
                if 0<no[r]<1:
                    count_ps+=1
                elif no[r]<0:
                    count_ns+=1
                elif no[r]==0.0:
                    count_unchanged+=1


    sev.append((("iou_positive", count_ps), ("iou_negative",count_ns ), ("iou_unchanged",count_unchanged)))

    return sev



if __name__ == '__main__':



    view = ["s40_n_cam_far", "s40_n_cam_near", "s50_s_cam_far", "s50_s_cam_near"]

    cam_name = "s40_n_cam_far"

    folder2 = cam_name + "_detections_yolov7_ids_new"
    gt_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder2)

    folder = cam_name + "_detections_yolov7_ids_try"
    detection_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder)

    folder = cam_name + "_detections_yolov7_ids_try_0"
    detection_new_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder)

    folder7 = cam_name + "_images"
    image_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder7)

    bb_det, bb_gt, _, stamps = load(detection_path, gt_path)
    bb_det1, bb_gt, _, stamps2 = load(detection_new_path, gt_path)

    counter_det, counter_undet,unmatched_groundtruth, matched_pairs, ids,ct_undet_dt = associate_GT_DT(bb_det, bb_gt, detection_path, gt_path)
    counter_det1, counter_undet1,unmatched_groundtruth1, matched_pairs1, ids1,ct_undet_dt1 = associate_GT_DT(bb_det1, bb_gt, detection_new_path, gt_path)

    nb_fp =  ct_undet_dt1 #unmatched detections; added but they should nt be added
    nb_fn = counter_undet1 #unmatched groundtruth; werent added but should be added
    nb_tp = counter_det1 #matched; added and should be added

    recall = nb_tp / (nb_tp + nb_fn)
    precision = nb_tp /(nb_tp + nb_fp)
    f1_score = (2 * precision * recall) / (precision + recall)

    print(f"recall_{cam_name} {recall}")
    print(f"precision_{cam_name} {precision}")
    print(f" f1_score_{cam_name} {f1_score}")

    _, _, _, _, merged_obj, trunc_obj, _,_,_ = merged(bb_gt, detection_path,gt_path,image_path,matched_pairs,stamps,ids)
    _, _, _, _, merged_obj1, trunc_obj1, _,_,_ = merged(bb_gt, detection_new_path,gt_path,image_path,matched_pairs1,stamps2,ids1)




    merge, trunc = iou_GT(merged_obj, trunc_obj)
    merge1, trunc1 = iou_GT(merged_obj1, trunc_obj1)

    iou_gt_corr = iou_GT1(matched_pairs1,gt_path)


    delta_iou_merge, delta_iou_trunc, ct_objs_false_from_beg, ct_objs_merg_from_beg = delta_GT(merge, trunc, merge1, trunc1,iou_gt_corr)

    print(f"Old trunc_{cam_name}  {ct_objs_false_from_beg}")
    print(f"Old merge_{cam_name}  {ct_objs_merg_from_beg}")

    sev = gather_ious(delta_iou_trunc)

    plot_hist_one(sev)


