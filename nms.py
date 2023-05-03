def iou(bbox1, bbox2):
    x1_b1, x2_b1, x1_b2, x2_b2 = bbox1[0], bbox2[0], bbox1[0] + bbox1[3], bbox2[0] + bbox2[3]
    y1_b1, y2_b1, y1_b2, y2_b2 = bbox1[1], bbox2[1], bbox1[1] + bbox1[4], bbox2[1] + bbox2[4]
    x_overlap = max(0, min(x1_b2, x2_b2)-max(x1_b1, x2_b1))
    y_overlap = max(0, min(y1_b2, y2_b2)-max(y1_b1, y2_b1))
    overlap_area = x_overlap * y_overlap
    area_1 = bbox1[3] * bbox2[4]
    area_2 = bbox2[3] * bbox2[4]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)

def nms(detections, threshold=.5):
    if len(detections) == 0:
        return []
    detections = sorted(detections, key=lambda detections: detections[2],reverse=True)
    new_detections=[]
    new_detections.append(detections[0])
    del detections[0]
    for index, detection in enumerate(detections):
        flag = True
        for new_detection in new_detections:
            if iou(detection, new_detection) > threshold:
                flag = False
                break
        if flag:
            new_detections.append(detection)
    return new_detections
