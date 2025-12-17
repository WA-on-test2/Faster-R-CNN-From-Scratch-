import torch


class AnchorBoxGenerator:
    def __init__(self, base_sizes, aspect_ratios):
      
        self.base_anchor_sizes = base_sizes
        self.anchor_aspect_ratios = aspect_ratios
        self.anchors_per_location = len(base_sizes) * len(aspect_ratios)
    
    def generate_anchors_for_feature_map(self, input_tensor, feature_tensor):
     
        feature_h, feature_w = feature_tensor.shape[-2:]#Generates anchor boxes at each position of the feature map
        input_h, input_w = input_tensor.shape[-2:]     
        stride_h = torch.tensor(input_h // feature_h, dtype=torch.int64, 
                               device=feature_tensor.device)
        stride_w = torch.tensor(input_w // feature_w, dtype=torch.int64, 
                               device=feature_tensor.device)
        
        sizes_tensor = torch.as_tensor(self.base_anchor_sizes, 
                                       dtype=feature_tensor.dtype,
                                       device=feature_tensor.device)
        ratios_tensor = torch.as_tensor(self.anchor_aspect_ratios,
                                        dtype=feature_tensor.dtype,
                                        device=feature_tensor.device)
        
        # Compute anchor dimensions
        # For area = size^2 and aspect_ratio = h/w:
        # h = sqrt(area * ratio), w = sqrt(area / ratio)
        height_multipliers = torch.sqrt(ratios_tensor)
        width_multipliers = 1 / height_multipliers
        
        widths = (width_multipliers[:, None] * sizes_tensor[None, :]).view(-1)#actual
        heights = (height_multipliers[:, None] * sizes_tensor[None, :]).view(-1)
        
        
        base_anchor_set = torch.stack([-widths, -heights, widths, heights], dim=1) / 2#centered at zero
        base_anchor_set = base_anchor_set.round()
        
        x_shifts = torch.arange(0, feature_w, dtype=torch.int32, #shifts
                               device=feature_tensor.device) * stride_w
        y_shifts = torch.arange(0, feature_h, dtype=torch.int32,
                               device=feature_tensor.device) * stride_h
        
        shift_y, shift_x = torch.meshgrid(y_shifts, x_shifts, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        
        shift_vectors = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
        
        # Broadcasting: (H*W, 1, 4) + (1, num_anchors, 4) = (H*W, num_anchors, 4)
        all_anchors = (shift_vectors.view(-1, 1, 4) + #apply shifts
                      base_anchor_set.view(1, -1, 4))
        all_anchors = all_anchors.reshape(-1, 4) # Reshape to (H*W*num_anchors, 4)
        
        return all_anchors