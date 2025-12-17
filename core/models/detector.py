"""
Combines backbone, RPN, and ROI head into complete detector
"""

import torch
import torch.nn as nn
import torchvision

from core.models.region_proposal import ProposalGenerationNetwork
from core.models.roi_classifier import RegionClassificationHead
from utils.bbox_operations import rescale_boxes_to_original


class ObjectDetectionModel(nn.Module):
 
    def __init__(self, network_config, total_classes):
        super(ObjectDetectionModel, self).__init__()
        
        self.network_config = network_config
        
        # Build backbone (VGG16)
        pretrained_vgg = torchvision.models.vgg16(pretrained=True)
        self.feature_extractor = pretrained_vgg.features[:-1]
        
        # Initialize RPN
        self.proposal_network = ProposalGenerationNetwork(
            network_config['backbone_out_channels'],
            scales=network_config['scales'],
            anchor_ratios=network_config['aspect_ratios'],
            network_config=network_config)
        
        # Initialize ROI head
        self.classification_head = RegionClassificationHead(
            network_config, 
            total_classes,
            in_channels=network_config['backbone_out_channels'])
        
        # Freeze early backbone layers
        for layer_module in self.feature_extractor[:10]:
            for parameter in layer_module.parameters():
                parameter.requires_grad = False
        
        # Normalization parameters (ImageNet stats)
        self.normalization_mean = [0.485, 0.456, 0.406]
        self.normalization_std = [0.229, 0.224, 0.225]
        
        # Image resizing parameters
        self.target_min_dimension = network_config['min_im_size']
        self.target_max_dimension = network_config['max_im_size']
    
    def _preprocess_image_and_boxes(self, input_image, bounding_boxes):
   
        data_type, computation_device = input_image.dtype, input_image.device
        
        mean_tensor = torch.as_tensor(self.normalization_mean, 
                                     dtype=data_type, device=computation_device)
        std_tensor = torch.as_tensor(self.normalization_std,
                                     dtype=data_type, device=computation_device)
        normalized_image = (input_image - mean_tensor[:, None, None]) / std_tensor[:, None, None]
        
        original_h, original_w = input_image.shape[-2:]
        dimension_tensor = torch.tensor(input_image.shape[-2:])
        smaller_dimension = torch.min(dimension_tensor).to(dtype=torch.float32)
        larger_dimension = torch.max(dimension_tensor).to(dtype=torch.float32)
        
        scale_factor = torch.min(
            float(self.target_min_dimension) / smaller_dimension,
            float(self.target_max_dimension) / larger_dimension
        )
        scale_multiplier = scale_factor.item()
        
        resized_image = torch.nn.functional.interpolate(
            normalized_image,
            size=None,
            scale_factor=scale_multiplier,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )
        
        if bounding_boxes is not None:
            scale_ratios = [
                torch.tensor(new_dim, dtype=torch.float32, device=bounding_boxes.device)
                / torch.tensor(old_dim, dtype=torch.float32, device=bounding_boxes.device)
                for new_dim, old_dim in zip(resized_image.shape[-2:], (original_h, original_w))
            ]
            
            height_ratio, width_ratio = scale_ratios
            x_min, y_min, x_max, y_max = bounding_boxes.unbind(2)
            
            x_min = x_min * width_ratio
            x_max = x_max * width_ratio
            y_min = y_min * height_ratio
            y_max = y_max * height_ratio
            
            adjusted_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=2)
        else:
            adjusted_boxes = None
        
        return resized_image, adjusted_boxes
    
    def forward(self, input_tensor, annotation_targets=None):
       
        original_dimensions = input_tensor.shape[-2:]
        
        if self.training:
            processed_image, adjusted_boxes = self._preprocess_image_and_boxes(
                input_tensor, annotation_targets['bboxes'])
            annotation_targets['bboxes'] = adjusted_boxes
        else:
            processed_image, _ = self._preprocess_image_and_boxes(input_tensor, None)
        
        extracted_features = self.feature_extractor(processed_image)
        
        # Generate region proposals
        rpn_results = self.proposal_network(processed_image, extracted_features, 
                                           annotation_targets)
        generated_proposals = rpn_results['proposals']
        
        detection_results = self.classification_head(
            extracted_features, 
            generated_proposals,
            processed_image.shape[-2:],
            annotation_targets)
        
        if not self.training:
            detection_results['boxes'] = rescale_boxes_to_original(
                detection_results['boxes'],
                processed_image.shape[-2:],
                original_dimensions)
        
        return rpn_results, detection_results