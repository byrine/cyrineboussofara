import numpy as np

from occlusion_utils.sort import *
from Loading_Visualization.json_files_edit import load_json_files_with_ids as load
from Loading_Visualization.draw_occ import *

from occlusion_utils.bbox_overlap_utils import *
import os
from drawing_package.drawing_package.data_interface.load_data import load_SensorOutput
from drawing_package.drawing_package.data_interface.save_data import save_SensorOutput
from scripts.save_as_table import *


def area(bb):
    bb_area = abs((bb[3] - bb[2]) * (bb[1] - bb[0]))

    return bb_area

def associate_GT_DT(bb_det,bb_gt,det_path, gt_path):




    counter_det = 0
    counter_undet = 0
    counter_undet_dt = 0
    matched_pairs = dict()
    unmatched_groundtruth = dict()
    ids = dict()
    for filename in os.listdir(det_path):


        m_pair= []
        unmatched = []
        unmatched_id = []
        name_det = det_path + filename
        name_gt = gt_path + filename
        matched_pairs[name_gt] = []
        ids[name_gt] = []
        unmatched_groundtruth[name_gt] = []
        bb_d = bb_det[name_det][0]
        bb_g = bb_gt[name_gt][0]


        bb_id = bb_gt[name_gt][1]
        bb_id_det = bb_det[name_det][1]
        iou_threshold = 0.3
        bb_g1 = convert_bbox_to_xyxy(bb_g)
        bb_d1 = convert_bbox_to_xyxy(bb_d)

        matched, unmatched_gt, unmatched_dets = associate_detections_to_trackers(bb_g1, bb_d1, iou_batch, iou_threshold)

        counter_det += len(matched)
        for m in matched:
            matched_pairs[name_gt].append((bb_g[m[0]],bb_d[m[1]]))
            ids[name_gt].append((bb_id[m[0]], bb_id_det[m[1]]))



        #matched_pairs[name_gt].append(m_pair)
        counter_undet_dt += len(unmatched_dets)
        counter_undet += len(unmatched_gt)
        for i in unmatched_gt:
            unmatched.append(bb_g[i])
            unmatched_id.append(bb_id[i])

        unmatched_groundtruth[name_gt].append(unmatched)
        unmatched_groundtruth[name_gt].append(unmatched_id)



    return counter_det,counter_undet,unmatched_groundtruth, matched_pairs,ids, counter_undet_dt


def check(bbxs, occlus_param,match,match_id,stamp,frame_nb):

    a = []
    c = []

    d = []
    e = []
    ct_undetected_occ = 0
    ct_both_undected_occ = 0
    ct_undetected_not_occ = 0
    ct_undetected_both_not_occ = 0
    ct_undetected_t = 0
    ct_undetected_f = 0
    ct_undetected_f1 = 0
    ct_undetected_si = 0
    ct_undetected_si1 = 0
    ct_undetected_e = 0
    ct_undetected_e1 = 0
    ct_undetected_c = 0
    ct_undetected_c1 = 0
    ct_undetected_s = 0
    ct_undetected_s1 = 0
    ct_undetected_n = 0
    ct_undetected_n1 = 0
    ct_undetected_t1 = 0
    ct_undetected_o = 0
    ct_undetected_o1 = 0
    ct_undetected_d1 = 0
    ct_undetected_d = 0
    m= []
    n=[]
    r = []
    t = []
    j=[]
    g = []
    l = []
    p = []
    q=[ ]
    mn = []
    x = []
    y =[]
    tt = []
    mm = []
    mz = []
    l1 = []
    a_det = []
    list_det_undet = []
    '''if len(bbxs) == 0:
        a = []
        c = []
        d = []
        e = []
        m = []
        n = []
        r = []
        t = []
        j = []
        g = []
        l = []
        p = []
        q = []
        mn = []
        x = []
        y = []
        tt = []
        mm = []
        mz = []
        l1 = []
        a_det = []

        return  (c,a, d, e,a_det, ct_undetected_occ ,ct_both_undected_occ ,ct_undetected_not_occ,ct_undetected_both_not_occ ,ct_undetected_t ,ct_undetected_f ,ct_undetected_f1 ,ct_undetected_si ,ct_undetected_si1 ,ct_undetected_e ,ct_undetected_e1 ,ct_undetected_c ,
            ct_undetected_c1 ,ct_undetected_s ,ct_undetected_s1 ,ct_undetected_n ,ct_undetected_n1 ,ct_undetected_t1 ,ct_undetected_o ,ct_undetected_o1 ,ct_undetected_d1 ,ct_undetected_d)
    '''
    lenOcc = len(occlus_param)

    for k in range(lenOcc):
        obj1 = occlus_param[k]
        lenA = len(obj1)
        for o in range(lenA):
            obj = obj1[o]


            obj3 = obj[8]
            obj2 = obj[9]

            if obj[0] :


                if 0.1<=obj[7]:

                    if len(bbxs) !=0 and obj3 in bbxs and obj3 not in a:
                        width_0 = obj[8][3] - obj[8][2]
                        height_0 = obj[8][1] - obj[8][0]

                        a.append([obj[11],-1,stamp, frame_nb,obj[7], obj[8],-1,width_0,height_0,-1,-1, 'yes'])
                        list_det_undet.append([obj[11],-1,stamp, frame_nb,obj[7], obj[8],-1,width_0,height_0,-1,-1, 'yes'])
                        ct_undetected_occ +=1

                    elif len(bbxs) !=0 and obj2 in bbxs and obj2 not in a :
                        width_0 = obj[9][3] - obj[9][2]
                        height_0 = obj[9][1] - obj[9][0]

                        a.append([obj[12],-1,stamp, frame_nb, obj[7], obj[9], -1,width_0,height_0,-1,-1,'yes'])
                        list_det_undet.append([obj[12],-1,stamp, frame_nb, obj[7], obj[9], -1,width_0,height_0,-1,-1,'yes'])
                        ct_undetected_occ += 1

                    else:
                        for box in match:
                            if obj3 == box[0] and obj3 not in a_det:
                                width_0 = obj[8][3] - obj[8][2]
                                height_0 = obj[8][1] - obj[8][0]
                                width_1 = box[1][3] - box[1][2]
                                height_1 = box[1][1] - box[1][0]

                                rank = match.index(box)
                                id_det = match_id[rank][1]

                                a_det.append(
                                        [obj[11], id_det, stamp, frame_nb, obj[7], obj[8], obj[8],width_0,height_0,
                                         width_1,height_1, 'no'])
                                list_det_undet.append(
                                        [obj[11], id_det, stamp, frame_nb, obj[7], obj[8], obj[8],width_0,height_0,
                                         width_1,height_1, 'no'])


                            elif obj2 == box[0] and  obj2 not in a_det:
                                width_0 = obj[9][3] - obj[9][2]
                                height_0 = obj[9][1] - obj[9][0]
                                width_1 = box[1][3] - box[1][2]
                                height_1 = box[1][1] - box[1][0]

                                rank = match.index(box)
                                id_det = match_id[rank][1]
                                a_det.append([obj[12], id_det, stamp, frame_nb, obj[7], obj[9], obj[9], width_0,height_0,
                                         width_1,height_1, 'no'])
                                list_det_undet.append([obj[12], id_det, stamp, frame_nb, obj[7], obj[9], obj[9], width_0,height_0,
                                         width_1,height_1, 'no'])
                            else :continue


                if 0.1 <= obj[7] < 0.2:

                    if obj3 in bbxs and obj3 not in m:
                        m.append([obj[7], obj[8]])
                        ct_undetected_o += 1
                    elif obj2 in bbxs and obj2 not in m:
                        m.append([obj[7], obj[9]])
                        ct_undetected_o += 1
                    elif obj3 not in bbxs and obj3 not in p:
                        p.append([obj[7], obj[8]])
                        ct_undetected_o1 += 1
                    elif obj2 not in bbxs and obj2 not in p:
                        p.append([obj[7], obj[9]])
                        ct_undetected_o1 += 1

                if 0.2 <= obj[7] < 0.3:

                    if obj3 in bbxs and not obj3 not in m:
                        m.append([obj[7], obj[8]])
                        ct_undetected_d += 1
                    elif obj2 in bbxs and obj2 not in m:
                        m.append([obj[7], obj[9]])
                        ct_undetected_d += 1
                    elif obj3 not in bbxs and obj3 not in q:
                        q.append([obj[7], obj[8]])
                        ct_undetected_d1 += 1
                    elif obj2 not in bbxs and obj2 not in q:
                        q.append([obj[7], obj[9]])
                        ct_undetected_d1 += 1

                if 0.3<=obj[7]<0.4 :

                    if obj3 in bbxs and obj3 not in m:
                        m.append([obj[7], obj[8]])
                        ct_undetected_t +=1
                    elif obj2 in bbxs and obj2 not in m:
                        m.append([obj[7], obj[9]])
                        ct_undetected_t += 1
                    elif obj3 not in bbxs and obj3 not in mn:
                        mn.append([obj[7], obj[8]])
                        ct_undetected_t1 +=1
                    elif obj2 not in bbxs and obj2 not in mn:
                        mn.append([obj[7], obj[9]])
                        ct_undetected_t1 += 1


                if 0.4<=obj[7]<0.5 :
                   if obj3 == bbxs and obj3 not in n:
                       n.append([obj[7], obj[8]])
                       ct_undetected_f += 1
                   elif obj2 == bbxs and obj2 not in n:
                       n.append([obj[7], obj[8]])
                       ct_undetected_f += 1
                   elif obj3 not in bbxs and obj2 not in x:
                       x.append([obj[7], obj[8]])
                       ct_undetected_f1 += 1
                   elif obj2 not in bbxs and obj2 not in x:
                       x.append([obj[7], obj[9]])
                       ct_undetected_f1 += 1

                if 0.5<=obj[7]<0.6 :

                    if  obj3 in bbxs and obj3 not in r:
                        r.append([obj[7], obj[8]])
                        ct_undetected_c += 1
                    elif obj2 in bbxs and obj2 not in r:
                        r.append([obj[7], obj[9]])
                        ct_undetected_c += 1
                    elif obj3 not in bbxs and obj2 not in mm:
                        mm.append([obj[7], obj[8]])
                        ct_undetected_c1 +=1
                    elif obj2 not in bbxs and obj2 not in mm:
                        mm.append([obj[7], obj[9]])
                        ct_undetected_c1 += 1

                if 0.6<=obj[7]<0.7 :
                    if obj3 in bbxs and obj3 not in g:
                        g.append([obj[7], obj[8]])
                        ct_undetected_si += 1
                    elif obj2 in bbxs and obj2 not in g:
                        g.append([obj[7], obj[9]])
                        ct_undetected_si += 1
                    elif obj3 not in bbxs and obj2 not in y:
                        y.append([obj[7], obj[8]])
                        ct_undetected_si1 +=1
                    elif obj2 not in bbxs and obj2 not in y:
                        y.append([obj[7], obj[9]])
                        ct_undetected_si1 += 1

                if 0.7 <= obj[7] < 0.8:
                    if  obj3 in bbxs and obj3 not in j:
                        j.append([obj[7], obj[8]])
                        ct_undetected_s += 1
                    elif obj2 in bbxs and obj2 not in j:
                        j.append([obj[7], obj[9]])
                        ct_undetected_s += 1
                    elif obj3 not in bbxs and obj2 not in tt:
                        tt.append([obj[7], obj[8]])
                        ct_undetected_s1 +=1
                    elif obj2 not in bbxs and obj2 not in tt:
                        tt.append([obj[7], obj[9]])
                        ct_undetected_s1 += 1

                if 0.8 <= obj[7] < 0.9:
                    if obj3 in bbxs and obj3 not in l:
                        l.append([obj[7], obj[8]])
                        ct_undetected_e += 1
                    elif obj2 in bbxs and obj2 not in l:
                        l.append([obj[7], obj[9]])
                        ct_undetected_e += 1
                    elif obj3 not in bbxs and obj2 not in l1:
                        l1.append([obj[7], obj[8]])
                        ct_undetected_e1 +=1
                    elif obj2 not in bbxs and obj2 not in l1:
                        l1.append([obj[7], obj[9]])
                        ct_undetected_e1 += 1

                if 0.9 <= obj[7] :
                    if obj3 in bbxs and obj3 not in t:
                        t.append([obj[7], obj[8]])
                        ct_undetected_n += 1
                    elif obj2 in bbxs and obj2 not in t:
                        t.append([obj[7], obj[8]])
                        ct_undetected_n += 1
                    elif obj3 not in bbxs and obj2 not in mz:
                        mz.append([obj[7], obj[8]])
                        ct_undetected_n1 +=1
                    elif obj2 not in bbxs and obj2 not in mz:
                        m.append([obj[7], obj[9]])
                        ct_undetected_n1 += 1


            else:


                if obj3 in bbxs and obj3 not in e:
                    e.append(obj[8])
                elif obj2 in bbxs and obj2 not in e:
                    e.append(obj[9])
                elif obj3 in bbxs and obj2 in bbxs and obj3 not in d and obj2 not in d and obj3 not in e and obj2 not in e :
                    ct_undetected_both_not_occ += 1
                    d.append([obj[8], obj[9]])

            for i in a:
                if i in e:
                    e.remove(i)

            ct_undetected_not_occ = len(e)
            ct_undetected_occ = len(a)


    return (c,a, d, e,a_det,list_det_undet, ct_undetected_occ ,ct_both_undected_occ ,ct_undetected_not_occ,ct_undetected_both_not_occ,ct_undetected_t ,ct_undetected_f ,ct_undetected_f1 ,ct_undetected_si ,ct_undetected_si1 ,ct_undetected_e ,ct_undetected_e1 ,ct_undetected_c ,
            ct_undetected_c1 ,ct_undetected_s ,ct_undetected_s1 ,ct_undetected_n ,ct_undetected_n1 ,ct_undetected_t1 ,ct_undetected_o ,ct_undetected_o1 ,ct_undetected_d1 ,ct_undetected_d)



def undetected_occ(bb_gt, det_path,gt_path,images_path, unmatch_gt,match_id,match,stamps):

    #_, bb_gt = load_data(det_path, gt_path)

    det_occ, im = detection_overlap(bb_gt, images_path, gt_path, 1)
    occlusion = find_occlusion(bb_gt, det_occ, images_path,gt_path )

    #lenList = len(occlusion)
    lala = os.listdir(gt_path)
    len_f = len(lala)
    list_undetected_both_occ = []
    list_undetected_occ =[]
    list_undetected_both_not_occ = []
    list_undetected_not_occ = []
    list_undetected_occ_det = []
    list_b = []
    ct_unde_occ = 0
    ct_undected_not_occ = 0
    ct_undected_b_not_occ = 0
    ct_undected_b_occ = 0
    ct_undetected_three = 0
    ct_undetected_five = 0
    ct_undetected_four = 0
    ct_undetected_six = 0
    ct_undetected_eight = 0
    ct_undetected_seven = 0
    ct_undetected_nine = 0
    ct_undetected_one = 0
    ct_undetected_two = 0
    ct_undetected_three1 = 0
    ct_undetected_five1 = 0
    ct_undetected_four1 = 0
    ct_undetected_six1 = 0
    ct_undetected_eight1 = 0
    ct_undetected_seven1 = 0
    ct_undetected_nine1 = 0
    ct_undetected_one1 = 0
    ct_undetected_two1 = 0
    sev_nb = []
    sev_nb1 = []
    for i in range(len_f):
        filename =  gt_path + lala[i]
        bbxs = unmatch_gt[filename][0]
        matcho_id = match_id[filename]
        matcho = match[filename]
        stamp = stamps[filename]
        nb_frame = i
        #bbxs = np.array(bbxs)

        occlus_param = occlusion[i]

        c, a, d, e, a_det, list_det_undet, ct_undet_occ, ct_b_undet_occ, ct_undet_not_occ, ct_undet_b_not_occ, three, four, four1,six, six1, eight, eight1, five,five1, seven, seven1, nine, nine1, three1,one, one1, two1, two = check(bbxs,occlus_param,matcho,matcho_id,stamp,nb_frame)
        list_b.append(list_det_undet)
        ct_unde_occ +=ct_undet_occ
        ct_undected_not_occ += ct_undet_not_occ
        ct_undected_b_occ += ct_b_undet_occ
        ct_undected_b_not_occ += ct_undet_b_not_occ
        ct_undetected_four += four
        ct_undetected_six += six
        ct_undetected_eight += eight
        list_undetected_both_occ.append(c)
        list_undetected_occ.append(a)
        list_undetected_both_not_occ.append(d)
        list_undetected_not_occ.append(e)
        ct_undetected_three += three
        ct_undetected_five += five
        ct_undetected_seven += seven
        ct_undetected_nine += nine
        ct_undetected_one += one
        ct_undetected_two += two
        ct_undetected_three1 += three1
        ct_undetected_five1 += five1
        ct_undetected_four1 += four1
        ct_undetected_six1 += six1
        ct_undetected_eight1 += eight1
        ct_undetected_seven1 += seven1
        ct_undetected_nine1 += nine1
        ct_undetected_one1 += one1
        ct_undetected_two1 += two1
        list_undetected_occ_det += a_det

    if ct_undetected_one != 0 or ct_undetected_one1 != 0:

        per_und_one = (ct_undetected_one * 100) /(ct_undetected_one + ct_undetected_one1 )
        per_d_one = (ct_undetected_one1 * 100) / (ct_undetected_one + ct_undetected_one1)
    else:
        per_und_one = 0
        per_d_one = 0

    if ct_undetected_two != 0 or ct_undetected_two1 != 0:
        per_und_two = (ct_undetected_two * 100) / (ct_undetected_two + ct_undetected_two1)
        per_d_two1 = (ct_undetected_two1 * 100) / (ct_undetected_two + ct_undetected_two1)
    else:
        per_und_two = 0
        per_d_two1 = 0

    if ct_undetected_three != 0 or ct_undetected_three1 != 0:
        per_und_three = (ct_undetected_three * 100) / (ct_undetected_three + ct_undetected_three1)
        per_d_three = (ct_undetected_three1 * 100) / (ct_undetected_three + ct_undetected_three1)
    else:
        per_und_three = 0
        per_d_three = 0

    if ct_undetected_four != 0 or ct_undetected_four1 != 0:
        per_und_four = (ct_undetected_four * 100) / (ct_undetected_four + ct_undetected_four1)
        per_d_four = (ct_undetected_four1 * 100) / (ct_undetected_four + ct_undetected_four1)
    else:
        per_und_four = 0
        per_d_four = 0

    if ct_undetected_five != 0 or ct_undetected_five1 != 0:
        per_und_five = (ct_undetected_five * 100) / (ct_undetected_five + ct_undetected_five1)
        per_d_five = (ct_undetected_five1 * 100) / (ct_undetected_five + ct_undetected_five1)
    else:
        per_und_five = 0
        per_d_five = 0

    if ct_undetected_six != 0 or ct_undetected_six1 != 0:
        per_und_six = (ct_undetected_six * 100) / (ct_undetected_six + ct_undetected_six1)
        per_d_six = (ct_undetected_six1 * 100) / (ct_undetected_six + ct_undetected_six1)
    else:
        per_und_six = 0
        per_d_six = 0


    if ct_undetected_seven != 0 or ct_undetected_seven1 != 0:
        per_und_seven = (ct_undetected_seven * 100) / (ct_undetected_seven + ct_undetected_seven1)
        per_d_seven = (ct_undetected_seven1 * 100) / (ct_undetected_seven + ct_undetected_seven1)
    else:
        per_und_seven = 0
        per_d_seven = 0

    if ct_undetected_eight != 0 or ct_undetected_eight1 != 0:
        per_und_eight = (ct_undetected_eight * 100) / (ct_undetected_eight + ct_undetected_eight1)
        per_d_eight = (ct_undetected_eight1 * 100) / (ct_undetected_eight + ct_undetected_eight1)
    else:
        per_und_eight = 0
        per_d_eight = 0

    if ct_undetected_nine != 0 or ct_undetected_nine1 != 0:

        per_und_nine = (ct_undetected_nine * 100) / (ct_undetected_nine + ct_undetected_nine1)
        per_d_nine = (ct_undetected_nine1 * 100) / (ct_undetected_nine + ct_undetected_nine1)
    else:
        per_und_nine = 0
        per_d_nine = 0

    sev_nb.append(((0.1,ct_undetected_one,per_und_one),(0.2,ct_undetected_two,per_und_two),(0.3,ct_undetected_three,per_und_three),(0.4,ct_undetected_four,per_und_four), (0.5,ct_undetected_five,per_und_five), (0.6,ct_undetected_six,per_und_six) ,(0.7,ct_undetected_seven,per_und_seven),(0.8,ct_undetected_eight,per_und_eight),(0.9,ct_undetected_nine,per_und_nine)))
    sev_nb1.append(((0.1,ct_undetected_one1,per_d_one),(0.2,ct_undetected_two1,per_d_two1),(0.3,ct_undetected_three1,per_d_three),(0.4,ct_undetected_four1,per_d_four), (0.5,ct_undetected_five1,per_d_five), (0.6,ct_undetected_six1,per_d_six) ,(0.7,ct_undetected_seven1,per_d_seven),(0.8,ct_undetected_eight1,per_d_eight),(0.9,ct_undetected_nine1,per_d_nine)))


    return list_b,list_undetected_occ_det,list_undetected_both_occ, list_undetected_occ, list_undetected_both_not_occ,list_undetected_not_occ,ct_unde_occ,ct_undected_not_occ,ct_undected_b_occ,ct_undected_b_not_occ,sev_nb,sev_nb1

def save_data(data_list,undected_occ, list_und_not_occ):

    for i in range(len(data_list)):

        list1_temp = []
        list2_temp = []

        undetected_w_occ = undected_occ[i]
        undec_not_occ = list_und_not_occ[i]
        check = data_list[i].object_list
        check1 =  data_list[i].object_list.copy()


        len_occ2 = len(undetected_w_occ)
        for r in range(len_occ2):
            list2_temp.append(undetected_w_occ[r][1])
        len_occ3 = len(undec_not_occ)
        counter = data_list[i].num_detected
        for obj in check:
            if len_occ2==0 and len_occ3 == 0:
                check1.clear()
                counter = 0


            elif len_occ2 != 0 or len_occ3 != 0:

                if obj.boundingbox in list2_temp or obj.boundingbox in undec_not_occ :
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

    bb_det,bb_gt,_, stamps1 = load(detection_path,gt_path)


    ct_det, ct_undet, unmatch_gt,match ,id, ct_undet_dt= associate_GT_DT(bb_det, bb_gt,detection_path, gt_path)
    print(f"nb of undetected, {ct_undet}")

    list_b,list_det_occ,_, list_undet_occ,_, list_und_not_occ, ct_unde_occ, ct_unde_n_occ, ct_undected_b_occ,_,sev_nb,sev_nb1=  undetected_occ(bb_gt,detection_path,gt_path,image_path,unmatch_gt,id,match,stamps1)
    #print(f"nb of undetected because of occ, {ct_unde_occ}")
    #print(f"nb of undetected not because of occ, {ct_unde_n_occ}")
    #print(f"nb of both undetected because of occ, {ct_undected_b_occ}")

    #table(list_b,gt_path)
    #plot_histo_sev(sev_nb,sev_nb1)

    load_dir = dir + '{}_detections_yolov7_ids_new'.format(cam_name)


    data = load_SensorOutput(load_dir + "/")
    data['data_list'] = save_data(data['data_list'],list_undet_occ,list_und_not_occ)

    save_dir = load_dir + "_undetected"
    #if not os.path.isdir(save_dir):
    #    os.mkdir(save_dir)
    #save_SensorOutput(data, save_dir + "/")



