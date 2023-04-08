from occlusion_utils.GTvsDT_undetected import *
from occlusion_utils.GTvsDT_merged_trun import *
from Loading_Visualization.draw_occ import *
from stats_gather.Metrics import *


def main():

    #cam_name = "s40_n_cam_far"
    #view = ["s40_n_cam_far","s40_n_cam_near", "s50_s_cam_far","s50_s_cam_near"]


    cam_name = "s40_n_cam_far"
    folder = cam_name + "_detections_yolov7_ids_try"
    detection_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder)

    folder2 = cam_name + "_detections_yolov7_ids_new"
    gt_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder2)

    folder1 = cam_name + "_detections_yolov7_ids_try_0"
    detection_new_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder1)


    folder7 = cam_name + "_images"
    image_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder7)

    bb_det,bb_gt,_, stamps1 = load(detection_path,gt_path)
    bb_det1,bb_gt,_, stamps2 = load(detection_new_path,gt_path)


    ct_det, ct_undet, unmatch_gt,match ,id,_= associate_GT_DT(bb_det, bb_gt,detection_path, gt_path)
    ct_det1, ct_undet1, unmatch_gt1,match1 ,id1,_= associate_GT_DT(bb_det1, bb_gt,detection_new_path, gt_path)

    #print(f"nb of undetected, {ct_undet}")

    list_b,_,_, _,_, _, ct_unde_occ, _, _,_,sev_nb,sev_nb1=  undetected_occ(bb_gt,detection_path,gt_path,image_path,unmatch_gt,id,match,stamps1)
    list_b1,_,_, _,_, _, ct_unde_occ1, _, _,_,sev_nb11,sev_nb111=  undetected_occ(bb_gt,detection_new_path,gt_path,image_path,unmatch_gt1,id1,match1,stamps2)

    det = []
    det1 = []
    save_path_und_not_corr = "C:/Users/cyrin/OneDrive/Documents/Masterarbeit/stats/{}/not_corr_data_occs_UndevsDet.csv".format(cam_name)
    save_path_corr = "C:/Users/cyrin/OneDrive/Documents/Masterarbeit/stats/{}/corr_data_occs_UndevsDet.csv".format(cam_name)

    #table(list_b,save_path_und_not_corr)
    #table(list_b1,save_path_corr)
    plot_histo_sev(sev_nb, sev_nb1)
    #plot_histo_sev(sev_nb11, sev_nb111)

    ct_merged_images, _, ct_trunc_images, _, merged_obj, trunc_obj, _,_,list_m_t = merged(bb_gt, detection_path,gt_path,image_path,match,stamps1,id)
    ct_merged_images1, _, ct_trunc_images1, _, merged_obj1, trunc_obj1, _,_,list_m_t1 = merged(bb_gt, detection_new_path,gt_path,image_path,match1,stamps2,id1)


    delta = ct_unde_occ-ct_unde_occ1

    merge, trunc = iou_GT(merged_obj, trunc_obj)
    merge1, trunc1 = iou_GT(merged_obj1, trunc_obj1)

    iou_gt_corr = iou_GT1(match1,gt_path)


    _, _, ct_objs_false_from_beg, ct_objs_merg_from_beg = delta_GT(merge, trunc, merge1, trunc1,iou_gt_corr)

    new_trunc = ct_trunc_images1-ct_objs_false_from_beg

    delta1 = ct_trunc_images-new_trunc
    delta2 = ct_merged_images - ct_merged_images1


    if (ct_unde_occ !=0 or ct_unde_occ1 !=0 ) and (ct_merged_images!= 0 or ct_merged_images1!=0) and (ct_trunc_images!=0 or ct_trunc_images1!=0 ):

        det.append((("undetected",ct_unde_occ),("merged",ct_merged_images),("truncated",ct_trunc_images)))
        det1.append((("undetected", ct_unde_occ1,- round( (delta / ct_unde_occ ) * 100)), ("merged", ct_merged_images1,- round( (delta2 / ct_merged_images ) * 100)), ("truncated", new_trunc,- round( (delta1 / ct_trunc_images ) * 100))))

    elif (ct_unde_occ ==0 and ct_unde_occ1 ==0 ):
        det.append((("undetected", ct_unde_occ), ("merged", ct_merged_images), ("truncated", ct_trunc_images)))
        det1.append((("undetected", ct_unde_occ1, 0), ("merged", ct_merged_images1, - round( (delta2 / ct_merged_images ) * 100)), ("truncated", new_trunc,- round( (delta1 / ct_trunc_images ) * 100))))
    elif (ct_merged_images ==0 and ct_merged_images1 ==0 ):
        det.append((("undetected", ct_unde_occ), ( "merged", ct_merged_images), ("truncated", ct_trunc_images)))
        det1.append((("undetected", ct_unde_occ1, - round( (delta / ct_unde_occ ) * 100)), ("merged", ct_merged_images1,0), ("truncated", new_trunc,  -round( (delta1 / ct_trunc_images ) * 100))))
    elif (ct_trunc_images ==0 and ct_trunc_images1 ==0 ):
        det.append((("undetected", ct_unde_occ),("merged", ct_merged_images), ("truncated", ct_trunc_images)))
        det1.append((("undetected", ct_unde_occ1, - round( (delta / ct_unde_occ ) * 100)),("merged", ct_merged_images1, - round( (delta2 / ct_merged_images ) * 100)), ("truncated", new_trunc, 0)))



    plot_histo_all(det1,det)

    save_path_und_not_corr1 = "C:/Users/cyrin/OneDrive/Documents/Masterarbeit/stats/{}/not_corr_data_occs_merge_trunc.csv".format(cam_name)
    save_path_corr1 = "C:/Users/cyrin/OneDrive/Documents/Masterarbeit/stats/{}/corr_data_occs_merge_trunc.csv".format(cam_name)



    #table(list_m_t,save_path_und_not_corr1)
    #table(list_m_t1,save_path_corr1)



    pie_chart(ct_unde_occ,ct_merged_images,ct_trunc_images)


if __name__ == '__main__':
    main()



