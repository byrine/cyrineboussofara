from drawing_package.drawing_package.data_interface.load_data import load_SensorOutput
from drawing_package.drawing_package.data_interface.save_data import save_SensorOutput
from occlusion_utils.sort import *
import numpy as np
import copy
from occlusion_utils.GTvsDT_merged_trun import *
folder = '2021-08-09-12-49-28'
dir = 'C:/Users/cyrin/{}/'.format(folder)

cams = ['s40_n_cam_far', 's40_n_cam_near', 's50_s_cam_near' , 's50_s_cam_far' ]



def dist_bb(bb1,bb2):

    dist = np.linalg.norm(bb1-bb2)


    return dist


def load_bb_id(data_list):
    bbox = []
    bb_ids = []
    bounding = []
    for i in range(len(data_list)):
        bb_image = []
        bb_id = []
        bound = []
        object_list = data_list[i].object_list
        for j in range(len(object_list)):
            obj = object_list[j]
            boxi = np.array(obj.boundingbox) [0:4]
            boxi = boxi.tolist()
            bb_image.append(tuple((obj, boxi, obj.object_id, obj.object_class)))
            #bb_image.append(obj)
            #bb_image.append(boxi)
            #bb_image.append(obj.object_id, obj.object_class)))
            #bb_id.append(obj.object_id)
            bound.append(boxi)

        bbox.append(bb_image)
        bounding.append(bound)
        bb_ids.append(bb_id)

    return bbox,bb_ids, bounding

def create_pred(box):#data_list):
    bb_pred = []
    #_, _ , bbox = load_bb_id(data_list)
    args = parse_args()
    #mot_tracker =  Sort_obj(max_age=args.max_age,
      #                     min_hits=args.min_hits,
      #                    nb_frames=args.nb_frames,
        #                   iou_threshold=args.iou_threshold)
    mot_tracker =  Sort(max_age=args.max_age,
                        min_hits=args.min_hits,
                       nb_frames=args.nb_frames,
                      iou_threshold=args.iou_threshold)

    '''
    for i in range(len(data_list)):
        box = bbox[i]
        box = np.array(box)[:,0:4]
        new_form = convert_bbox_to_xyxy(box)
        new_form = np.array(new_form)
        bb_float_pred = []
        trackers = mot_tracker.update(iou_batch, new_form)
        for d in trackers:

            bb_float_pred.append([d[1],d[3],d[0],d[2]])


        bb_pred.append(bb_float_pred)
    '''
    #box = bbox[i]
    box = np.array(box)[:,0:4]
    new_form = convert_bbox_to_xyxy(box)
    new_form = np.array(new_form)
    bb_float_pred = []
    trackers = mot_tracker.update(iou_batch, new_form)
    for d in trackers:

        bb_float_pred.append([d[1],d[3],d[0],d[2]])

    return bb_float_pred

def preprocess_add_pred( data_list):

    obj, bb_ids,bbox = load_bb_id(data_list)
    #bb_pred = create_pred(data_list)
    bb_p=create_pred(bbox[0])
    box_previous=obj[0]

    for i in range(1, len(data_list)):
        box = bbox[i]
        box = np.array(box)[:,0:4]
        #box_previous = obj[i-1]
        box_fut = obj[i]
        b = bbox[i-1]

        box_o = np.array(b)[:,0:4]

        box_o = box_o.tolist()
        box = box.tolist()
        id_data = bb_ids[i]
        #bb_p = bb_pred[i-1]
        counter = len(box)

        check = data_list[i].object_list
        new = copy.deepcopy(check)
        chek_previous = data_list[i-1].object_list
        box_old = convert_bbox_to_xyxy(box_o)
        box1 = convert_bbox_to_xyxy(box)
        bb_p1 = convert_bbox_to_xyxy(bb_p)
        es = []
        es_match_pred = []
        es_match = []
        iou_threshold = 0.3
        l_prev = len(box_previous)
        l_fut =  len(box_fut)
        list_fut = []
        if(i>2):
            b2 = bbox[i - 2]
        else: b2=b
        box_o_o = np.array(b2)[:, 0:4]

        box_o_o = box_o_o.tolist()

        #chek_previous = data_list[i - 1].object_list
        box_old_old = convert_bbox_to_xyxy(box_o_o)
        #box1 = convert_bbox_to_xyxy(box)
        bb_p1 = convert_bbox_to_xyxy(bb_p)

        matched, unmatched_dets, unmatched_preds = associate_detections_to_trackers(box1, bb_p1, iou_batch, iou_threshold)

        matched_old, unmatched_dets_old, unmatched_preds1 = associate_detections_to_trackers(box_old, bb_p1, iou_batch, iou_threshold)
        matched_old_old, unmatched_dets_old_old, unmatched_preds2 = associate_detections_to_trackers(box_old_old, bb_p1, iou_batch, iou_threshold)

        for r in unmatched_preds:
            es.append(bb_p[r])
        for r in matched:
            es_match_pred.append(bb_p[r[1]])
            es_match.append((box[r[0]], bb_p[r[1]]))
        es_match_old=[]
        es_match_old_old=[]

        for r in matched_old:
            es_match_old.append((box_o[r[0]]))

        for r in matched_old_old:
            es_match_old_old.append((box_o_o[r[0]]))

        counter =  data_list[i].num_detected

        for k in range(l_fut):
            #print("ffffffffffffffffffffffff")
            #print(check[k].position)
            '''
            for box in es_match:

                if box_fut[k][1] == box[0]:
                    #      check.remove(param2)
                    #     counter -= 1
                    b1 = box[1]
                    b0 = box[0]
                    if area(b0)  < area(b1)*0.95 or area(b0) > area(b1) * 2:
                        width=b1[3]-b1[2]
                        height=b1[1]-b1[0]
                        check[k].boundingbox = [b0[0],b0[0]+height,b0[2],b0[2]+width]
                        #check[k].boundingbox = b1
                    else: check[k].boundingbox = b0
            '''

            #print(check[k].position)
            lo=len(es_match)
            for y in range(lo):

                if box_fut[k][1] == es_match[y][0]:
                    #      check.remove(param2)
                    #     counter -= 1
                    b1 = es_match[y][1]
                    b0 = es_match[y][0]
                    if area(b0) < area(b1) * 0.95 or area(b0) > area(b1) *1.05 :
                        width = b1[3] - b1[2]
                        height = b1[1] - b1[0]
                        #check[k].boundingbox = [b0[0], b0[0] + height, b0[2], b0[2] + width]
                        check[k].boundingbox = b1
                    #else:
                        #check[k].boundingbox = b0


                    if area(b0) >  area(b1)*1.6:
                        if y<len(es_match_old):
                            for h in range(len(es_match_old)):
                                if np.allclose(b0, es_match_old[h], rtol=0, atol=0.009):
                                    if area(b0) > area(es_match_old[y]):
                                        if y<len(es_match_old_old):
                                            for o in range(len(es_match_old_old)):
                                                if np.allclose(es_match_old[h], es_match_old_old[o], rtol=0, atol=0.009):
                                                    if area(es_match_old[h])>area(es_match_old_old[o]):
                                                        check[k].boundingbox = b0
                                        #else: check[k].boundingbox = b0

                        #else: check[k].boundingbox = b0




            # print(check[k].position)
                #counter -= 1
        for k in range(l_prev):
            cla = box_previous[k][3]
            param1 = box_previous[k][0]
            pram = copy.deepcopy(param1)
            bd1 = box_previous[k][1]

            if cla == 2 :
                chek_previous.remove(param1)
                counter -= 1

            #bd1 = np.array(bd1)[0:4]
            #bd1 = bd1.tolist()
            verif = []

            #if len(unmatched_preds) != 0:

            '''for r in unmatched_preds:

                    predi = bb_p[r]
                    if np.allclose(predi,bd1, rtol=1e-01, atol=1e-01, equal_nan=False):
                        pram.boundingbox = predi
                        verif.append(pram.boundingbox)
                        check.append(pram)
                        counter +=1
            '''
            for m in matched_old:
                if bd1 == box_previous[m[0]][1]:

                        #if  bb_p[m[1]] in es or  bb_p[m[1]] in es_match_pred:
                        if bb_p[m[1]] in es:
                            pram.boundingbox = bb_p[m[1]]
                            check.append(pram)
                            counter += 1
                        #elif  bb_p[m[1]] in es_match_pred:

                            #for j in range(len(es_match)):

                               # pram.boundingbox = bb_p[m[1]]
                                #check.append(pram)
                                #counter += 1


        check_new=[]
        for c in check:
            check_new.append(c.boundingbox[0:4])
        bb_p=create_pred(check_new)
        box_previous = box_fut

        data_list[i].num_detected = counter


    return data_list



if __name__ == '__main__':

    #for cam in cams:
    #cam = "s40_n_cam_far"
    #view = ["s40_n_cam_far", "s40_n_cam_near", "s50_s_cam_far", "s50_s_cam_near"]

    #for cam in view:
    cam = "s40_n_cam_far"
    print("--------------", cam)
    load_dir = dir + '{}_detections_yolov7_ids_try'.format(cam)
    data = load_SensorOutput(load_dir + "/")

    #data['data_list']= preprocess_duplicate(data['data_list'])
    data['data_list'] = preprocess_add_pred(data['data_list'])

    save_dir = load_dir + "_0"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_SensorOutput(data, save_dir + "/",load_dir)


