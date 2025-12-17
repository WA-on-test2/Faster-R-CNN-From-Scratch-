
import torch
import torch.nn as nn
import torchvision

from core.layers.anchor_generator import AnchorBoxGenerator
from utils.bbox_operations import (
    calculate_intersection_over_union,
    encode_boxes_to_deltas,
    decode_deltas_to_boxes,
    constrain_boxes_to_boundaries
)
from utils.common import stratified_sample_instances


class ProposalGenerationNetwork(nn.Module):

    
    def __init__(self, input_channels, anchor_sizes, anchor_ratios, network_config):
        super(ProposalGenerationNetwork, self).__init__()
        
        self.anchor_generator = AnchorBoxGenerator(anchor_sizes, anchor_ratios)
        self.anchor_sizes = anchor_sizes
        self.background_threshold = network_config['rpn_bg_threshold']
        self.foreground_threshold = network_config['rpn_fg_threshold']
        self.nms_threshold = network_config['rpn_nms_threshold']
        
        self.training_batch_size = network_config['rpn_batch_size']
        self.foreground_fraction = int(network_config['rpn_pos_fraction'] * self.training_batch_size)
        
        self.max_proposals_after_nms = (network_config['rpn_train_topk'] if self.training 
                                       else network_config['rpn_test_topk'])
        self.max_proposals_before_nms = (network_config['rpn_train_prenms_topk'] if self.training
                                        else network_config['rpn_test_prenms_topk'])
        
        self.total_anchors_per_cell = self.anchor_generator.anchors_per_location
        self.feature_conv = nn.Conv2d(input_channels, input_channels, 
                                     kernel_size=3, stride=1, padding=1)
        self.objectness_head = nn.Conv2d(input_channels, self.total_anchors_per_cell,
                                        kernel_size=1, stride=1)
        self.regression_head = nn.Conv2d(input_channels, self.total_anchors_per_cell * 4,
                                        kernel_size=1, stride=1)
        
       
        for layer in [self.feature_conv, self.objectness_head, self.regression_head]: # Initialize weights
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
    
    def _match_anchors_to_targets(self, anchor_set, target_boxes):
        """
        Assign ground truth boxes to anchors based on IoU
        
        Args:
            anchor_set: All anchor boxes (N_anchors, 4)
            target_boxes: Ground truth boxes (N_gt, 4)
        
        Returns:
            labels: Classification labels (N_anchors) {-1/0/1}
            assigned_targets: Matched GT boxes (N_anchors, 4)
        """
        overlap_matrix = calculate_intersection_over_union(target_boxes, anchor_set) #  GT boxes and anchors
        best_overlap_per_anchor, best_gt_idx_per_anchor = overlap_matrix.max(dim=0) #best GT for each anchor
        
        # Save original assignments before thresholding
        original_assignments = best_gt_idx_per_anchor.clone()
        
        # Apply thresholds to create initial labels
        low_quality_mask = best_overlap_per_anchor < self.background_threshold
        medium_quality_mask = ((best_overlap_per_anchor >= self.background_threshold) & 
                              (best_overlap_per_anchor < self.foreground_threshold))
        
        best_gt_idx_per_anchor[low_quality_mask] = -1
        best_gt_idx_per_anchor[medium_quality_mask] = -2
        
        # Add highest IoU matches for each GT (even if below threshold)
        best_anchor_per_gt, _ = overlap_matrix.max(dim=1)
        
        # Find all anchor-GT pairs with maximum IoU for that GT
        highest_overlap_pairs = torch.where(overlap_matrix == best_anchor_per_gt[:, None])
        anchor_indices_to_force = highest_overlap_pairs[1]
        
        # Restore original GT assignments for these anchors
        best_gt_idx_per_anchor[anchor_indices_to_force] = original_assignments[anchor_indices_to_force]
        
        # Get matched GT boxes (clamp to handle background/ignored anchors)
        assigned_target_boxes = target_boxes[best_gt_idx_per_anchor.clamp(min=0)]
        
        # Create label tensor
        anchor_labels = (best_gt_idx_per_anchor >= 0).to(dtype=torch.float32)
        anchor_labels[low_quality_mask] = 0.0
        anchor_labels[medium_quality_mask] = -1.0
        
        return anchor_labels, assigned_target_boxes
    
    def _apply_proposal_filters(self, proposal_set, objectness_logits, canvas_shape):
        """
        Filter proposals using NMS and size constraints
        
        Args:
            proposal_set: Proposed boxes (N, 4)
            objectness_logits: Objectness scores (N,)
            canvas_shape: Image dimensions for clamping
        
        Returns:
            Filtered proposals and scores
        """
        # Convert to probabilities
        objectness_logits = objectness_logits.reshape(-1)
        objectness_probs = torch.sigmoid(objectness_logits)
        
        # Pre-NMS top-k selection
        _, top_indices = objectness_probs.topk(
            min(self.max_proposals_before_nms, len(objectness_probs)))
        
        objectness_probs = objectness_probs[top_indices]
        proposal_set = proposal_set[top_indices]
        
        # Clamp to image boundaries
        proposal_set = constrain_boxes_to_boundaries(proposal_set, canvas_shape)
        
        # Remove small boxes
        min_dimension = 16
        box_widths = proposal_set[:, 2] - proposal_set[:, 0]
        box_heights = proposal_set[:, 3] - proposal_set[:, 1]
        size_filter = (box_widths >= min_dimension) & (box_heights >= min_dimension)
        valid_indices = torch.where(size_filter)[0]
        
        proposal_set = proposal_set[valid_indices]
        objectness_probs = objectness_probs[valid_indices]
        
        # Apply NMS
        survivor_mask = torch.zeros_like(objectness_probs, dtype=torch.bool)
        nms_survivors = torch.ops.torchvision.nms(proposal_set, objectness_probs, 
                                                  self.nms_threshold)
        survivor_mask[nms_survivors] = True
        survivor_indices = torch.where(survivor_mask)[0]
        
        # Sort by objectness and take top-k
        sorted_by_score = survivor_indices[objectness_probs[survivor_indices].sort(descending=True)[1]]
        final_proposals = proposal_set[sorted_by_score[:self.max_proposals_after_nms]]
        final_scores = objectness_probs[sorted_by_score[:self.max_proposals_after_nms]]
        
        return final_proposals, final_scores
    
    def forward(self, input_image, feature_map, ground_truth=None):
        """
        Forward pass through RPN
        
        Args:
            input_image: Input image tensor (N, C, H, W)
            feature_map: Feature map from backbone (N, C_feat, H_feat, W_feat)
            ground_truth: Training targets (optional)
        
        Returns:
            Dictionary containing proposals, scores, and losses (if training)
        """
        # Apply RPN convolutions
        intermediate_features = nn.ReLU()(self.feature_conv(feature_map))
        objectness_predictions = self.objectness_head(intermediate_features)
        delta_predictions = self.regression_head(intermediate_features)
        
        # Generate anchor boxes
        anchor_boxes = self.anchor_generator.generate_anchors_for_feature_map(
            input_image, feature_map)
        
        # Reshape predictions
        num_anchors_per_cell = objectness_predictions.size(1)
        
        # Reshape objectness: (B, A, H, W) -> (B*H*W*A, 1)
        objectness_predictions = objectness_predictions.permute(0, 2, 3, 1)
        objectness_predictions = objectness_predictions.reshape(-1, 1)
        
        # Reshape deltas: (B, A*4, H, W) -> (B*H*W*A, 4)
        delta_predictions = delta_predictions.view(
            delta_predictions.size(0),
            num_anchors_per_cell,
            4,
            intermediate_features.shape[-2],
            intermediate_features.shape[-1])
        delta_predictions = delta_predictions.permute(0, 3, 4, 1, 2)
        delta_predictions = delta_predictions.reshape(-1, 4)
        
        # Decode anchors to proposals
        decoded_proposals = decode_deltas_to_boxes(
            delta_predictions.detach().reshape(-1, 1, 4),
            anchor_boxes)
        decoded_proposals = decoded_proposals.reshape(decoded_proposals.size(0), 4)
        
        # Filter proposals
        filtered_proposals, proposal_scores = self._apply_proposal_filters(
            decoded_proposals, objectness_predictions.detach(), input_image.shape)
        
        network_output = {
            'proposals': filtered_proposals,
            'scores': proposal_scores
        }
        
        if self.training and ground_truth is not None:
            anchor_labels, matched_boxes = self._match_anchors_to_targets(
                anchor_boxes,
                ground_truth['bboxes'][0])
            
            regression_targets = encode_boxes_to_deltas(matched_boxes, anchor_boxes)
            
            bg_sample_mask, fg_sample_mask = stratified_sample_instances(
                anchor_labels,
                foreground_target=self.foreground_fraction,
                batch_capacity=self.training_batch_size)
            
            sampled_mask = fg_sample_mask | bg_sample_mask
            sampled_indices = torch.where(sampled_mask)[0]
            
            # Compute localization loss (only on positives)
            bbox_loss = (
                torch.nn.functional.smooth_l1_loss(
                    delta_predictions[fg_sample_mask],
                    regression_targets[fg_sample_mask],
                    beta=1 / 9,
                    reduction="sum",
                ) / sampled_indices.numel()
            )
            
            # Compute classification loss
            classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                objectness_predictions[sampled_indices].flatten(),
                anchor_labels[sampled_indices].flatten())
            
            network_output['rpn_classification_loss'] = classification_loss
            network_output['rpn_localization_loss'] = bbox_loss
        
        return network_output