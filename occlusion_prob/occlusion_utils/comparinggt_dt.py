def find_corresponding_bb(ids,id, bb):
    rank =  ids.index(id)
    bb_co = bb[rank]

    return bb_co

def area_bb(bb):
    area= abs((bb[3] - bb[2]) * (bb[1] - bb[0]))

    return area



def categorize_occlusion(occlusions_GT, occlusions_DT,object_ids_dt,object_ids_gt,detection_path):
    lenList = len(occlusions_GT)

    trunc_odee_image=dict()
    trunc_oded_image = dict()
    merg_occluded = dict()
    not_det_odee = dict()
    not_det_oded = dict()


    nb_occludee_truncated = 0
    nb_occluded_truncated = 0

    nb_occluded_merge =  0

    nb_not_det_odee = 0
    nb_not_det_oded = 0

    for i in range(lenList):


        occgt_im = occlusions_GT[i]
        occdt_im = occlusions_DT[i]
        filename = list[i]
        trunc_odee_image[filename] = []
        trunc_oded_image[filename] = []
        merg_occluded[filename] = []
        not_det_odee[filename] = []
        not_det_oded[filename] = []


        filename = detection_path + filename
        obj_id = object_ids_dt[filename][1]
        bbox = object_ids_dt[filename][0]
        obj_id_gt = object_ids_dt[filename][1]
        bbox_gt = object_ids_dt[filename][0]


        lenOcc = len(occgt_im)
        lenOcc_DT = len(occdt_im)


        for k in range(lenOcc):
            obj1 = occgt_im[k]
            lenA = len(obj1)
            for o in range(lenA):
                obj = obj1[o]
                if obj[0]:
                    if obj[8] in obj_id and obj[8]==obj[6]: # the object was also detected and is occludee in GT
                        bb_dt_id = find_corresponding_bb(obj_id,obj[8],bbox)
                        bb_gt_id = find_corresponding_bb(obj_id_gt, obj[8], bbox_gt)

                        bb_dt_area = area_bb(bb_dt_id)
                        bb_gt_area = area_bb(bb_gt_id)


                        if bb_gt_area>bb_dt_area:
                            nb_occludee_truncated+=1
                            trunc_odee_image[filename].append(obj[8])


                    if obj[8] in obj_id and obj[8] == obj[5]:  # the object was also detected and is occluder in GT
                        bb_dt_id = find_corresponding_bb(obj_id, obj[8], bbox)
                        bb_gt_id = find_corresponding_bb(obj_id_gt, obj[8], bbox_gt)

                        bb_dt_area = area_bb(bb_dt_id)
                        bb_gt_area = area_bb(bb_gt_id)

                        if bb_gt_area > bb_dt_area:
                            nb_occluded_truncated += 1
                            trunc_oded_image[filename].append(obj[8])
                        elif bb_dt_area > bb_gt_area:
                            nb_occluded_merge += 1
                            merg_occluded[filename].append([obj[8],obj[7]])

                    if  obj[8] not in obj_id and obj[8]==obj[6]: #the object was not detected and is occludee in GT

                        nb_not_det_odee += 1
                        not_det_odee[filename].append([obj[8],obj[7]])


                    if obj[8] not in obj_id and obj[8] == obj[5]:  # the object was not detected and is occluder in GT
                        nb_not_det_oded += 1
                        not_det_oded[filename].append(obj[8])

































