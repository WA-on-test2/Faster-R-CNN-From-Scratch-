import torch
import math


def calculate_intersection_over_union(rect_set_a, rect_set_b):
    area_a = (rect_set_a[:, 2] - rect_set_a[:, 0]) * (rect_set_a[:, 3] - rect_set_a[:, 1])
    area_b = (rect_set_b[:, 2] - rect_set_b[:, 0]) * (rect_set_b[:, 3] - rect_set_b[:, 1])
    left_x = torch.max(rect_set_a[:, None, 0], rect_set_b[:, 0])
    top_y = torch.max(rect_set_a[:, None, 1], rect_set_b[:, 1])
    right_x = torch.min(rect_set_a[:, None, 2], rect_set_b[:, 2])
    bottom_y = torch.min(rect_set_a[:, None, 3], rect_set_b[:, 3])
    overlap_area = (right_x - left_x).clamp(min=0) * (bottom_y - top_y).clamp(min=0)
    combined_area = area_a[:, None] + area_b - overlap_area
    intersection_over_union = overlap_area / combined_area
    
    return intersection_over_union


def encode_boxes_to_deltas(reference_rects, target_rects):
    ref_widths = reference_rects[:, 2] - reference_rects[:, 0]
    ref_heights = reference_rects[:, 3] - reference_rects[:, 1]
    ref_center_x = reference_rects[:, 0] + 0.5 * ref_widths
    ref_center_y = reference_rects[:, 1] + 0.5 * ref_heights
    tgt_widths = target_rects[:, 2] - target_rects[:, 0]
    tgt_heights = target_rects[:, 3] - target_rects[:, 1]
    tgt_center_x = target_rects[:, 0] + 0.5 * tgt_widths
    tgt_center_y = target_rects[:, 1] + 0.5 * tgt_heights
    delta_x = (tgt_center_x - ref_center_x) / ref_widths
    delta_y = (tgt_center_y - ref_center_y) / ref_heights
    delta_w = torch.log(tgt_widths / ref_widths)
    delta_h = torch.log(tgt_heights / ref_heights)
    
    encoded_deltas = torch.stack((delta_x, delta_y, delta_w, delta_h), dim=1)
    return encoded_deltas


def decode_deltas_to_boxes(predicted_deltas, reference_rects):
    predicted_deltas = predicted_deltas.reshape(predicted_deltas.size(0), -1, 4)
    ref_w = reference_rects[:, 2] - reference_rects[:, 0]
    ref_h = reference_rects[:, 3] - reference_rects[:, 1]
    ref_cx = reference_rects[:, 0] + 0.5 * ref_w
    ref_cy = reference_rects[:, 1] + 0.5 * ref_h
    dx = predicted_deltas[..., 0]
    dy = predicted_deltas[..., 1]
    dw = predicted_deltas[..., 2]
    dh = predicted_deltas[..., 3]
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))
    pred_cx = dx * ref_w[:, None] + ref_cx[:, None]
    pred_cy = dy * ref_h[:, None] + ref_cy[:, None]
    pred_w = torch.exp(dw) * ref_w[:, None]
    pred_h = torch.exp(dh) * ref_h[:, None]
    pred_x1 = pred_cx - 0.5 * pred_w
    pred_y1 = pred_cy - 0.5 * pred_h
    pred_x2 = pred_cx + 0.5 * pred_w
    pred_y2 = pred_cy + 0.5 * pred_h
    
    decoded_rects = torch.stack((pred_x1, pred_y1, pred_x2, pred_y2), dim=2)
    return decoded_rects


def constrain_boxes_to_boundaries(rectangles, canvas_dimensions):
    x1_coords = rectangles[..., 0]
    y1_coords = rectangles[..., 1]
    x2_coords = rectangles[..., 2]
    y2_coords = rectangles[..., 3]
    
    canvas_h, canvas_w = canvas_dimensions[-2:]
    
    x1_coords = x1_coords.clamp(min=0, max=canvas_w)
    x2_coords = x2_coords.clamp(min=0, max=canvas_w)
    y1_coords = y1_coords.clamp(min=0, max=canvas_h)
    y2_coords = y2_coords.clamp(min=0, max=canvas_h)
    
    constrained_rects = torch.cat((
        x1_coords[..., None],
        y1_coords[..., None],
        x2_coords[..., None],
        y2_coords[..., None]),
        dim=-1)
    
    return constrained_rects


def rescale_boxes_to_original(rectangles, scaled_dims, original_dims):
 
    scale_ratios = [
        torch.tensor(orig_s, dtype=torch.float32, device=rectangles.device)
        / torch.tensor(curr_s, dtype=torch.float32, device=rectangles.device)
        for curr_s, orig_s in zip(scaled_dims, original_dims)
    ]
    
    ratio_h, ratio_w = scale_ratios
    x_min, y_min, x_max, y_max = rectangles.unbind(1)
    
    x_min = x_min * ratio_w
    x_max = x_max * ratio_w
    y_min = y_min * ratio_h
    y_max = y_max * ratio_h
    
    return torch.stack((x_min, y_min, x_max, y_max), dim=1)