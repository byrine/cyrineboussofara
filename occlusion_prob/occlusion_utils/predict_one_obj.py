from occlusion_utils.bbox_overlap_utils import *
from Loading_Visualization.json_files_edit import *



def main(detection_path,images_path, bboxes):

    for filename in os.listdir(detection_path):
        a = []
        image_file = images_path + "image_" + filename[5:9] + ".jpg"  ##path of an image
        det_path = detection_path + filename
        seq = np.array(bboxes[det_path][0])
        seq = seq[0:4]
        print(f"seq {bboxes[det_path][0]}")
        h = 1200
        w = 1920
        img = cv2.imread(image_file)
        y_top, y_bottom, x_left, x_right = predict_occlusion_1obj(seq)
        print(f"seq {[y_top,y_bottom,x_left,x_right]}")

        x_left_orig = int((seq[2] * w))
        y_top_orig = int((seq[0] * h))
        x_right_orig = int((seq[3] * w))
        y_bottom_orig = int((seq[1] * h))

        x_left = int((x_left * w))
        y_top = int((y_top * h))
        x_right= int((x_right * w))
        y_bottom = int((y_bottom * h))

        cv2.rectangle(img, (x_left, y_top), (x_right, y_bottom), (0, 0, 255), 2)
        cv2.rectangle(img, (x_left_orig, y_top_orig), (x_right_orig, y_bottom_orig), (255, 0, 0), 2)


        cv2.imshow("Img", img)
        cv2.waitKey(0)



if __name__ == '__main__':
    cam_name = "s40_n_cam_far"

    folder = cam_name + "_detections_yolov7_ids"
    detection_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder)

    folder2 = cam_name + "_images"
    images_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder2)

    folder3 = cam_name + "_det_occ"
    occlusion_path = 'C:/Users/cyrin/2021-08-09-12-49-28/{}/'.format(folder3)

    bbox, _, _, _ = load_json_files_dic_of_all_bbox_of_all_images(detection_path)


    main(detection_path,images_path, bbox)









