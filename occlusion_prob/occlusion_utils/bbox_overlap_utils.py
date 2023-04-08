import json
import os
import numpy as np
from occlusion_utils.centroid_tracker import CentroidTracker
import cv2
from occlusion_utils.kalmanfilter import KalmanFilter


def get_occlusion(bb1, bb2, i, j, r):
    """
    Gives all occlusion parameters back (yes/no, bbox coordinates,occluder/occluded,severity)

    Parameters
    ----------
    bb1 : list
        [y_top, y_bottom, x_left, x_right]
    bb2 : list
         [y_top, y_bottom, x_left, x_right]
    i: id of object with bb1
    j: id of object with bb2

    Returns
    -------
    boolean, floats, int, float

    """

    # determine the coordinates of the intersection rectangle

    x_left = max(bb1[2], bb2[2])
    y_top = max(bb1[0], bb2[0])
    x_right = min(bb1[3], bb2[3])
    y_bottom = min(bb1[1], bb2[1])

    occlusion = True

    if x_right < x_left or y_bottom < y_top:
        occlusion = False
        x_top_left = 0.0
        y_top_left = 0.0
        x_bottom_right = 0.0
        y_bottom_right = 0.0


        return occlusion, x_top_left, y_top_left, x_bottom_right, y_bottom_right, 0, 0, 0.0, bb1, bb2,False, i,j

    ##top-left point of intersection rectangle
    x_top_left = x_left
    y_top_left = y_top

    ###bottom_right point of intersection rectangle
    x_bottom_right = x_right
    y_bottom_right = y_bottom

    intersection_area = (x_bottom_right - x_top_left) * (y_bottom_right - y_top_left)

    bb1_area = abs((bb1[3] - bb1[2]) * (bb1[1] - bb1[0]))
    bb2_area = abs((bb2[3] - bb2[2]) * (bb2[1] - bb2[0]))

    # occluded/occluder
    occluder_first = True
    occluder = i
    occluded = j

    # occlusion severity: how much an object is occluded by another object at the moment
    #occ_severity = intersection_area / bb2_area

    occ_severity1 = intersection_area / bb1_area
    occ_severity2 = intersection_area / bb2_area

    occ_severity=max(occ_severity1,occ_severity2)*r

    # y_top<y_bottom #y_top1<y_top2
    if bb1[0] < bb2[1] and bb1[0] < bb2[0]:
        occluder_first= False
        occluder = j  # second object
        occluded = i  # first object
        #occ_severity = intersection_area / bb1_area
    bb1 = bb1.tolist()
    bb2 = bb2.tolist()

    return occlusion, x_top_left, y_top_left, x_bottom_right, y_bottom_right, occluder, occluded, occ_severity,bb1,bb2,occluder_first, i,j


def detection_overlap(dic_objectIds, images_path, detection_path,r):
    """
    Detects occlusions for each object in each image
    Gives back a list containing the results of the detection for each object in each image

    Parameters
    ----------
    bboxes: detected boxes in every image
    Returns
    -------
    list

    """

    occlusion_params_per_image = []
    img_nb = []

    for keys in dic_objectIds.keys():
        k = 0
        # bb = np.array(bboxes[keys])
        # bb = bb[:, 0:4]
        occlusion_params = []
        bb = np.array(dic_objectIds[keys][0])
        len_path = len(keys)
        if bb.any():
            if len_path== 74:
                img_nb.append(keys[65:69])

            else: img_nb.append(keys[66:70])

            bb = bb[:, 0:4]
            list = dic_objectIds[keys][1]
            lenB = len(list)
        else: continue

        for i in range(lenB):
            occ_bbi = []
            id1 = list[i]
            for j in range(lenB):
                k += 1
                id2 = list[j]
                if i < j:
                    occ_bbi.append(get_occlusion(bb[i], bb[j], id1, id2,r))
            occlusion_params.append(occ_bbi)
        occlusion_params_per_image.append(occlusion_params)

    return occlusion_params_per_image, img_nb

def find_occlusion(bboxes, detection, images_path, detection_path):
    """
        appends only the occlusions for each image if they exist and their parameters

        Parameters
        ----------
        bboxes: detected boxes in every image
        detection_path: path of detected objects
        Returns
        -------
        list

        """
    #print(detection[4])
    coordinate_occlusion_per_image = []
    lenDetection = len(detection)
    for k in range(lenDetection):
        det = detection[k]
        coordinate_occlusion = []
        lenDet = len(det)
        for i in range(lenDet):
            a = []
            deti = det[i]
            lenDeti = len(deti)
            for j in range(lenDeti):
                occ = deti[j]
                occlusion = occ[0]
                if occlusion:
                    val = occ[7]
                    if  val>=1.0:
                        val = 0.99

                    a.append([occlusion, occ[1], occ[2], occ[3], occ[4], occ[5], occ[6], val, occ[8],occ[9],occ[10],occ[11],occ[12]])
            if a:
                coordinate_occlusion.append(a)
        if coordinate_occlusion:
            coordinate_occlusion_per_image.append(coordinate_occlusion)
        else:
            coordinate_occlusion_per_image.append([[[False, 0.0, 0.0, 0.0, 0.0, -1, -1, 0.0, occ[8],occ[9],occ[10],occ[11],occ[12]]]])

    return coordinate_occlusion_per_image


id_obj_per_image = dict()


def object_id_non_cons(detection_path):
    for filename in os.listdir(detection_path):
        filename = detection_path + filename

        id_obj_per_image[filename] = []
        id = 0

        with open(filename) as jsondata:
            json_array = json.load(jsondata)
        id_obj_list = []
        for x in json_array["object_list"]:
            id_obj_per_image[filename].append(id)
            id += 1

    return id_obj_per_image


def predict_occlusion_one_frame(bboxes, id_obj, detection_path,im,cst):
    """
       Predicts occlusion one frame at a time, gives back probability of occlusion in next frame

       Parameters
       ----------
       bboxes: detected boxes in every image
       id_obj: detected boxes and their ids
       detection_path: path of the detected objects
       Returns
       -------
       list

       """



    occlusion_pred_per_image = []

    for filename in os.listdir(detection_path):

        r = filename[5:9]
        if r in im:
            det_path = detection_path + filename
        else: continue
        # y_top, y_bottom, x_left_ x_right
        id_obje = id_obj[det_path]
        id = id_obje[1]

        bb = np.array(bboxes[det_path])
        bb = bb[:, 0:4]

        occl_prob = []
        len_ = len(bb)
        bbox_id_list = id_obje[0]
        len__ = len(bbox_id_list)


        for i in range(len_):
            occ_bi = []

            for r in range(len__):
                if np.array_equal(bbox_id_list[r], bb[i]):
                    indice = id[r]

            for j in range(len_):

                for r in range(len__):
                    if np.array_equal(bbox_id_list[r], bb[j]):
                        indicej = id[r]
                if i < j:

                    '''bb1_x = bb[i][3] - bb[i][2]
                    bb1_y= bb[i][1] - bb[i][0]

                    bb2_x = bb[j][3] - bb[j][2]
                    bb2_y = bb[j][1] - bb[j][0]

                    bb1_array= np.array((bb1_x,bb1_y))
                    bb2_array= np.array((bb2_x,bb2_y))
                    '''
                    #cst = 0.002 for near
                    #cst = 0.01 for far
                    tay = 0.0
                    tax = 0.0
                    width_i = bb[i][3] - bb[i][2]
                    height_i = bb[i][1] - bb[i][0]
                    width_j = bb[j][3] - bb[j][2]
                    height_j = bb[j][1] - bb[j][0]
                    # if (bb[i][0] - bb[j][1] < 0.05 and  bb[i][1] > bb[j][0] )  or (bb[j][0] - bb[i][1] < 0.05 and  bb[j][1] > bb[i][0] ):

                    if (bb[i][0] - bb[j][1] < cst and bb[i][1] > bb[j][0]):
                        tay = -(bb[i][0] - bb[j][0]) / (height_j + cst) + 1
                    if (tay > 1 and bb[j][0] - bb[i][1] < cst and bb[j][1] > bb[i][0]):
                        tay = -(bb[j][0] - bb[i][0]) / (height_i + cst) + 1
                    if (bb[i][2] - bb[j][3] < cst and bb[i][3] > bb[j][2]):

                        tax = -(bb[i][2] - bb[j][2]) / (width_j + cst) + 1
                    if (tax > 1 and bb[j][2] - bb[i][3] < cst and bb[j][3] > bb[i][2]):

                        tax = -(bb[j][2] - bb[i][2]) / (width_i + cst) + 1

                    if (bb[i][2] > bb[j][2] and bb[i][3] < bb[j][3]) or (bb[j][2] > bb[i][2] and bb[j][3] < bb[i][3]):
                        tax = 1.0
                    if bb[i][0] > bb[j][0] and bb[i][1] < bb[j][1] or bb[j][0] > bb[i][0] and bb[j][1] < bb[i][1]:
                        tay = 1.0

                    '''dist = np.linalg.norm(bb1_array-bb2_array)

                    if dist == 0.0:
                        dist_ = 1


                    else: dist_ = dist_ref/dist'''
                    prob = (tax * tay)/2

                    occ_bi.append([prob,indice, indicej])
            occl_prob.append(occ_bi)
        occlusion_pred_per_image.append(occl_prob)
    return occlusion_pred_per_image


def object_id_cons(bboxes, images_path, detection_path):
    """
         Assigns consistent ids to the objects
         Parameters
         ----------
         bboxes: detected boxes in every image
         images_path: path of the images
         detection_path: path of the detected objects
         Returns
         -------
         dictionary:
         ---keys: specific detection path
         ---values: list of lists of object ids and corresponding bounding boxes

         """

    tracker = CentroidTracker(6, 0.1)
    object_ids_images = dict()

    for filename in os.listdir(detection_path):

        img_path = images_path + "image_" + filename[5:10] + "jpg"
        det_path = detection_path + filename
        object_ids_images[det_path] = []
        img = cv2.imread(img_path)
        bb = np.array(bboxes[det_path])
        bb = bb[:, 0:4]

        objects, disap = tracker.update(bb)
        object_ids = []
        bbox1 = []
        for (objectId, bbox) in objects.items():
            if disap[objectId] == 0:
                object_ids.append(objectId)
                bbox1.append(bbox)
                y_top, y_bottom, x_left, x_right = bbox
                dh = 1200
                dw = 1920

                x_left = int((bbox[2] * dw))
                y_top = int((bbox[0] * dh))
                x_right = int((bbox[3] * dw))
                y_bottom = int((bbox[1] * dh))

                # cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (0, 0, 255), 2)
                # text = "ID: {}".format(objectId)
                # cv2.putText(img, text, (x_left, y_top - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        object_ids_images[det_path].append(bbox1)
        object_ids_images[det_path].append(object_ids)

        # cv2.imshow("Img", img)
        # cv2.waitKey(0)
    return object_ids_images


def det_path_generator(detection_path, fileRank):
    """
             gives back a specific path (of the previous data) depending on filerank value

             Parameters
             ----------
             fileRank: int
             detection_path: path of the detected objects
             Returns
             -------
             str

             """

    if (fileRank < 0):
        fileRank = 0
    det_path_previous = detection_path + "data_"
    if (fileRank < 10):
        det_path_previous += "000" + str(fileRank) + ".json"
    elif (fileRank < 100):
        det_path_previous += "00" + str(fileRank) + ".json"
    elif (fileRank < 1000):
        det_path_previous += "0" + str(fileRank) + ".json"
    else:
        det_path_previous += str(fileRank) + ".json"
    return det_path_previous


def predict_occlusion_sequence(id_obj, bboxes, images_path, detection_path):
    """
             Gives back the predicted bounding boxes with a kalmanfilter that:
             ---learns first how the objects move by looking at the previous frames
              10 for s40 and 5 for s50
             ---predicts bounding boxes for the future frame

             Parameters
             ----------
             id_obj: dict
             bboxes: detected boxes in every image
             images_path: path of the images
             detection_path: path of the detected objects
             Returns
             -------
             dictionary:
             ---keys: specific detection path
             ---values: list of the predicted bounding boxes

             """

    h = 1200
    w = 1920
    bbox_pred_images = dict()

    for filename in os.listdir(detection_path):
        print(filename)
        bb_float = []
        bb_ids = []

        img_path = images_path + "image_" + filename[5:9] + ".jpg"
        det_path = detection_path + filename

        bbox_pred_images[det_path] = []

        img = cv2.imread(img_path)

        bb = np.array(bboxes[det_path])
        bb = bb[:, 0:4]
        l = len(bb)
        # print(f"bb {bb}")
        id_obje = id_obj[det_path]
        id = id_obje[1]
        bbox_id_list = id_obje[0]
        len_ = len(bbox_id_list)

        prev_cons = 5
        fileRank = int(filename[5:9]) - prev_cons

        for i in range(l):
            track1 = KalmanFilter()
            track2 = KalmanFilter()
            bbox_pred_float = []

            for r in range(len_):
                if np.array_equal(bbox_id_list[r], bb[i]):
                    indice = id[r]

            for k in range(0, prev_cons - 1):
                det_path_previous = det_path_generator(detection_path, fileRank + k)
                id_obje_previous = id_obj[det_path_previous]
                ids_previous = id_obje_previous[1]
                if indice in ids_previous:
                    rank = ids_previous.index(indice)
                else:
                    continue
                bb_id = id_obje_previous[0][rank]

                track1.predict(bb_id[2], bb_id[0])
                track2.predict(bb_id[3], bb_id[1])

            x_left, y_top = track1.predict(bb[i][2], bb[i][0])
            x_right, y_bottom = track2.predict(bb[i][3], bb[i][1])

            x_left_orig = int((bb[i][2] * w))
            y_top_orig = int((bb[i][0] * h))
            x_right_orig = int((bb[i][3] * w))
            y_bottom_orig = int((bb[i][1] * h))

            bbox_pred_float.append((y_top / 1200) / 1.001)
            bbox_pred_float.append((y_bottom / 1200) * 1.001)
            bbox_pred_float.append((x_left / 1920) / 1.001)
            bbox_pred_float.append((x_right / 1920) * 1.001)

            cv2.rectangle(img, (x_left_orig, y_top_orig), (x_right_orig, y_bottom_orig), (0, 0, 255), 2)
            text = "ID: {}".format(indice)
            # cv2.putText(img, text, (x_left_orig, y_top_orig - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (255, 0, 0), 2)
            cv2.putText(img, text, (x_left, y_top - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)

            bb_float.append(bbox_pred_float)
            bb_ids.append(indice)

        bbox_pred_images[det_path].append(bb_float)
        bbox_pred_images[det_path].append(bb_ids)


        #cv2.imshow("Img", img)
        key = cv2.waitKey(0)

        if key == 27:
            break

    return bbox_pred_images

