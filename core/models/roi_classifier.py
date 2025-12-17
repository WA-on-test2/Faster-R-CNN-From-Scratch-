import torch
import torch.nn as nn
import torchvision

from utils.bbox_operations import (
    calculate_intersection_over_union,
    encode_boxes_to_deltas,
    decode_deltas_to_boxes,
    constrain_boxes_to_boundaries
)
from utils.common import stratified_sample_instances


class RegionClassificationHead(nn.Module):

    def __init__(self, network_config, total_classes, in_channels):
        super(RegionClassificationHead, self).__init__()
        
        self.class_count = total_classes
        self.training_batch_capacity = network_config['roi_batch_size']
        self.foreground_target_ratio = int(network_config['roi_pos_fraction'] * 
                                          self.training_batch_capacity)
        self.overlap_threshold_positive = network_config['roi_iou_threshold']
        self.overlap_threshold_negative = network_config['roi_low_bg_iou']
        self.suppression_threshold = network_config['roi_nms_threshold']
        self.max_detections_output = network_config['roi_topk_detections']
        self.min_confidence_threshold = network_config['roi_score_threshold']
        self.pooling_resolution = network_config['roi_pool_size']
        self.hidden_dimension = network_config['fc_inner_dim']
        
        flattened_input_size = in_channels * self.pooling_resolution * self.pooling_resolution#FC
        self.first_fc = nn.Linear(flattened_input_size, self.hidden_dimension)
        self.second_fc = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.classification_fc = nn.Linear(self.hidden_dimension, self.class_count)
        self.regression_fc = nn.Linear(self.hidden_dimension, self.class_count * 4)
        
        torch.nn.init.normal_(self.classification_fc.weight, std=0.01)
        torch.nn.init.constant_(self.classification_fc.bias, 0)
        torch.nn.init.normal_(self.regression_fc.weight, std=0.001)
        torch.nn.init.constant_(self.regression_fc.bias, 0)
    
    def _assign_proposals_to_targets(self, proposal_boxes, target_boxes, target_labels):#(N_proposals, 4),(N_gt, 4)
        overlap_matrix = calculate_intersection_over_union(target_boxes, proposal_boxes)
        best_overlap, best_gt_index = overlap_matrix.max(dim=0)
        background_mask = ((best_overlap < self.overlap_threshold_positive) & 
                          (best_overlap >= self.overlap_threshold_negative))
        ignored_mask = best_overlap < self.overlap_threshold_negative
        
        best_gt_index[background_mask] = -1
        best_gt_index[ignored_mask] = -2
        
        matched_target_boxes = target_boxes[best_gt_index.clamp(min=0)]
        
        proposal_labels = target_labels[best_gt_index.clamp(min=0)]
        proposal_labels = proposal_labels.to(dtype=torch.int64)
        
        proposal_labels[background_mask] = 0  # Background class
        proposal_labels[ignored_mask] = -1    # Ignore
        
        return proposal_labels, matched_target_boxes
    
    def _filter_detections(self, predicted_boxes, predicted_labels, predicted_scores):
       
        confidence_filter = torch.where(predicted_scores > self.min_confidence_threshold)[0]
        predicted_boxes = predicted_boxes[confidence_filter]
        predicted_scores = predicted_scores[confidence_filter]
        predicted_labels = predicted_labels[confidence_filter]
        
        min_box_size = 16
        box_widths = predicted_boxes[:, 2] - predicted_boxes[:, 0]
        box_heights = predicted_boxes[:, 3] - predicted_boxes[:, 1]
        size_filter = (box_widths >= min_box_size) & (box_heights >= min_box_size)
        valid_size_indices = torch.where(size_filter)[0]
        
        predicted_boxes = predicted_boxes[valid_size_indices]
        predicted_scores = predicted_scores[valid_size_indices]
        predicted_labels = predicted_labels[valid_size_indices]
        
        survivor_mask = torch.zeros_like(predicted_scores, dtype=torch.bool)
        
        for class_idx in torch.unique(predicted_labels):
            class_mask = torch.where(predicted_labels == class_idx)[0]
            class_survivors = torch.ops.torchvision.nms(
                predicted_boxes[class_mask],
                predicted_scores[class_mask],
                self.suppression_threshold)
            survivor_mask[class_mask[class_survivors]] = True
        
        survivor_indices = torch.where(survivor_mask)[0]
        
        sorted_indices = survivor_indices[predicted_scores[survivor_indices].sort(descending=True)[1]]
        top_k_indices = sorted_indices[:self.max_detections_output]
        
        predicted_boxes = predicted_boxes[top_k_indices]
        predicted_scores = predicted_scores[top_k_indices]
        predicted_labels = predicted_labels[top_k_indices]
        
        return predicted_boxes, predicted_labels, predicted_scores
    
    def forward(self, feature_map, proposal_boxes, canvas_dimensions, training_targets):
   
        if self.training and training_targets is not None:
            proposal_boxes = torch.cat([proposal_boxes, training_targets['bboxes'][0]], dim=0)
            
            gt_boxes = training_targets['bboxes'][0]
            gt_labels = training_targets['labels'][0]
            
            assigned_labels, assigned_boxes = self._assign_proposals_to_targets(
                proposal_boxes, gt_boxes, gt_labels)
            
            # Sample positive and negative proposals
            bg_mask, fg_mask = stratified_sample_instances(
                assigned_labels,
                foreground_target=self.foreground_target_ratio,
                batch_capacity=self.training_batch_capacity)
            
            sample_mask = fg_mask | bg_mask
            sampled_indices = torch.where(sample_mask)[0]
            
            proposal_boxes = proposal_boxes[sampled_indices]
            assigned_labels = assigned_labels[sampled_indices]
            assigned_boxes = assigned_boxes[sampled_indices]
            
            bbox_regression_targets = encode_boxes_to_deltas(assigned_boxes, proposal_boxes)
        
        feature_spatial_dims = feature_map.shape[-2:]
        spatial_scales = []
        
        for feat_dim, img_dim in zip(feature_spatial_dims, canvas_dimensions):
            approximate_scale = float(feat_dim) / float(img_dim)
            quantized_scale = 2 ** float(torch.tensor(approximate_scale).log2().round())
            spatial_scales.append(quantized_scale)
        
        assert spatial_scales[0] == spatial_scales[1], "Scales must match"
        
        pooled_features = torchvision.ops.roi_pool(
            feature_map, [proposal_boxes],
            output_size=self.pooling_resolution,
            spatial_scale=spatial_scales[0])
        
        pooled_features = pooled_features.flatten(start_dim=1)
        
        # Pass through fully connected layers
        fc1_output = torch.nn.functional.relu(self.first_fc(pooled_features))
        fc2_output = torch.nn.functional.relu(self.second_fc(fc1_output))
        
        classification_logits = self.classification_fc(fc2_output)
        regression_predictions = self.regression_fc(fc2_output)
        
        num_proposals, num_classes = classification_logits.shape
        regression_predictions = regression_predictions.reshape(num_proposals, num_classes, 4)
        
        output_dict = {}
        
        if self.training and training_targets is not None:
            cls_loss = torch.nn.functional.cross_entropy(classification_logits, assigned_labels)
            
            foreground_indices = torch.where(assigned_labels > 0)[0]
            foreground_class_labels = assigned_labels[foreground_indices]
            
            bbox_loss = torch.nn.functional.smooth_l1_loss(
                regression_predictions[foreground_indices, foreground_class_labels],
                bbox_regression_targets[foreground_indices],
                beta=1/9,
                reduction="sum",
            )
            bbox_loss = bbox_loss / assigned_labels.numel()
            
            output_dict['frcnn_classification_loss'] = cls_loss
            output_dict['frcnn_localization_loss'] = bbox_loss
        
        if not self.training:
            device = classification_logits.device
            
            final_boxes = decode_deltas_to_boxes(regression_predictions, proposal_boxes)
            final_scores = torch.nn.functional.softmax(classification_logits, dim=-1)
            
            final_boxes = constrain_boxes_to_boundaries(final_boxes, canvas_dimensions)
            
            label_tensor = torch.arange(num_classes, device=device)
            label_tensor = label_tensor.view(1, -1).expand_as(final_scores)
            
            # Remove background predictions
            final_boxes = final_boxes[:, 1:]
            final_scores = final_scores[:, 1:]
            label_tensor = label_tensor[:, 1:]
            
            final_boxes = final_boxes.reshape(-1, 4)
            final_scores = final_scores.reshape(-1)
            label_tensor = label_tensor.reshape(-1)
            
            final_boxes, label_tensor, final_scores = self._filter_detections(
                final_boxes, label_tensor, final_scores)
            
            output_dict['boxes'] = final_boxes
            output_dict['scores'] = final_scores
            output_dict['labels'] = label_tensor
        
        return output_dict