import numpy as np
import cv2



def draw_voc_bboxes(image, default_box, conf, loc):
    voc_color_map = [[0, 0, 0], [128, 0, 0], [0, 128, 0],
                     [128, 128, 0], [0, 0, 128], [128, 0, 128],
                     [0, 128, 128], [128, 128, 128], [64, 0, 0],
                     [192, 0, 0], [64, 128, 0], [192, 128, 0],
                     [64, 0, 128], [192, 0, 128], [64, 128, 128],
                     [192, 128, 128], [0, 64, 0], [128, 64, 0],
                     [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    default_box_cxy = default_box[:, :2] 
    default_box_wh = default_box[:, 2:]
    loc_cxy = loc[:, :2]
    loc_wh = loc[:, 2:]
    bbox_cxy = loc_cxy * 0.1 * default_box_wh + default_box_cxy
    bbox_wh = np.exp(loc_wh * 0.2) * default_box_wh
    mask = np.max(conf[:, 1:], -1) > 0.5
    labels = np.argmax(conf, -1)[mask]
    bbox_cxy = bbox_cxy[mask]
    bbox_wh = bbox_wh[mask]
    bbox_tl = bbox_cxy - bbox_wh / 2
    bbox_br = bbox_cxy + bbox_wh / 2
    bboxes = np.concatenate([bbox_tl, bbox_br], -1)
    height, width = image.shape[:2]
    for box, label in zip(bboxes, labels):
        x_min = int(box[0] * width)
        y_min = int(box[1] * height)
        x_max = int(box[2] * width)
        y_max = int(box[3] * height)
        color = voc_color_map[label][::-1]
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                              color, 1)
#        cv2.imshow('test', image)
#        cv2.waitKey(1)
    """
    default_box_wh = default_box_wh[mask]
    default_box_cxy = default_box_cxy[mask]
    default_box_tl = default_box_cxy - default_box_wh / 2
    default_box_br = default_box_cxy + default_box_wh / 2
    bboxes = np.concatenate([default_box_tl, default_box_br], -1)
    for box in bboxes:
        x_min = int(box[0] * width)
        y_min = int(box[1] * height)
        x_max = int(box[2] * width)
        y_max = int(box[3] * height)
        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max),
                              (0, 255, 0), 1)
    """
    return image




def test():
    label_map = {'background' : 0,
                 'aeroplane' : 1,
                 'bicycle' : 2,
                 'bird' : 3,
                 'boat' : 4,
                 'bottle' : 5,
                 'bus' : 6,
                 'car' : 7,
                 'cat' : 8,
                 'chair' : 9,
                 'cow' : 10,
                 'diningtable' : 11,
                 'dog' : 12,
                 'horse' : 13,
                 'motorbike' : 14,
                 'person' : 15,
                 'pottedplant' : 16,
                 'sheep' : 17,
                 'sofa' : 18,
                 'train' : 19,
                 'tvmonitor' : 20}
    voc_color_map = [[0, 0, 0], [128, 0, 0], [0, 128, 0],
                     [128, 128, 0], [0, 0, 128], [128, 0, 128],
                     [0, 128, 128], [128, 128, 128], [64, 0, 0],
                     [192, 0, 0], [64, 128, 0], [192, 128, 0],
                     [64, 0, 128], [192, 0, 128], [64, 128, 128],
                     [192, 128, 128], [0, 64, 0], [128, 64, 0],
                     [0, 192, 0], [128, 192, 0], [0, 64, 128]]
    pascal_voc_spec = [['id', 'name', 'color_b', 'color_g', 'color_r']]
    for name, number in label_map.items():
        color = voc_color_map[number][::-1]
        pascal_voc_spec.append([number, 
                                name,
                                color[2],
                                color[1],
                                color[0]])
    import csv
    with open('./pascal_voc_spec.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(pascal_voc_spec)
        


if __name__ == '__main__':
    test()
    