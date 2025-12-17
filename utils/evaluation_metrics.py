import numpy as np


def compute_single_iou(detection_rect, groundtruth_rect):
 
    det_x1, det_y1, det_x2, det_y2 = detection_rect
    gt_x1, gt_y1, gt_x2, gt_y2 = groundtruth_rect
    
    intersect_x1 = max(det_x1, gt_x1)
    intersect_y1 = max(det_y1, gt_y1)
    intersect_x2 = min(det_x2, gt_x2)
    intersect_y2 = min(det_y2, gt_y2)
    
    if intersect_x2 < intersect_x1 or intersect_y2 < intersect_y1:
        return 0.0
    
    intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    
    union_area = float(det_area + gt_area - intersect_area + 1e-6)
    iou_score = intersect_area / union_area
    
    return iou_score


def calculate_mean_average_precision(predictions_per_image, groundtruths_per_image, 
                                    overlap_threshold=0.5, aggregation_method='area'):

    all_class_names = {cls for img_gt in groundtruths_per_image for cls in img_gt.keys()}
    all_class_names = sorted(all_class_names)
    
    per_class_precisions = {}
    collected_precisions = []
    
    for class_idx, class_name in enumerate(all_class_names):
        class_detections = [
            [img_idx, detection]
            for img_idx, img_preds in enumerate(predictions_per_image)
            if class_name in img_preds
            for detection in img_preds[class_name]
        ]
        
        class_detections = sorted(class_detections, key=lambda x: -x[1][-1])
        gt_assignment_flags = [[False for _ in img_gts[class_name]] 
                              for img_gts in groundtruths_per_image]
        
        total_gt_count = sum([len(img_gts[class_name]) 
                             for img_gts in groundtruths_per_image])
    
        true_positives = [0] * len(class_detections)
        false_positives = [0] * len(class_detections)
        for det_idx, (img_idx, detection) in enumerate(class_detections):
            img_gts = groundtruths_per_image[img_idx][class_name]
            
            best_overlap = -1
            best_gt_idx = -1
            
            # best match
            for gt_idx, gt_box in enumerate(img_gts):
                overlap = compute_single_iou(detection[:-1], gt_box)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_gt_idx = gt_idx
            

            if best_overlap < overlap_threshold or gt_assignment_flags[img_idx][best_gt_idx]:
                false_positives[det_idx] = 1
            else:
                true_positives[det_idx] = 1
                gt_assignment_flags[img_idx][best_gt_idx] = True

        cumulative_tp = np.cumsum(true_positives)
        cumulative_fp = np.cumsum(false_positives)
        epsilon = np.finfo(np.float32).eps
        recall_values = cumulative_tp / np.maximum(total_gt_count, epsilon)
        precision_values = cumulative_tp / np.maximum((cumulative_tp + cumulative_fp), epsilon)
        
        if aggregation_method == 'area':
            ap_score = _compute_ap_by_area(recall_values, precision_values)
        elif aggregation_method == 'interp':
            ap_score = _compute_ap_by_interpolation(recall_values, precision_values)
        else:
            raise ValueError('aggregation_method must be "area" or "interp"')
        
        if total_gt_count > 0:
            collected_precisions.append(ap_score)
            per_class_precisions[class_name] = ap_score
        else:
            per_class_precisions[class_name] = np.nan
    
    mean_precision = sum(collected_precisions) / len(collected_precisions) if collected_precisions else 0.0
    
    return mean_precision, per_class_precisions


def _compute_ap_by_area(recall_array, precision_array):

    recall_array = np.concatenate(([0.0], recall_array, [1.0]))
    precision_array = np.concatenate(([0.0], precision_array, [0.0]))
    for idx in range(precision_array.size - 1, 0, -1):
        precision_array[idx - 1] = np.maximum(precision_array[idx - 1], precision_array[idx])
    change_indices = np.where(recall_array[1:] != recall_array[:-1])[0]
    ap_score = np.sum((recall_array[change_indices + 1] - recall_array[change_indices]) * 
                     precision_array[change_indices + 1])
    
    return ap_score


def _compute_ap_by_interpolation(recall_array, precision_array):
    ap_score = 0.0
    
    for interpolation_point in np.arange(0, 1 + 1e-3, 0.1):
        # recalls >= interpolation point
        valid_precisions = precision_array[recall_array >= interpolation_point]
        max_precision = valid_precisions.max() if valid_precisions.size > 0 else 0.0
        ap_score += max_precision
    
    ap_score = ap_score / 11.0
    return ap_score