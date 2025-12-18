import torch
from tqdm import tqdm
import numpy as np
from utils.common import get_computation_device
from utils.evaluation_metrics import calculate_mean_average_precision

class ModelEvaluator:
    def __init__(self, model, test_loader, dataset):
        self.model = model
        self.test_loader = test_loader
        self.dataset = dataset
        self.device = get_computation_device()
    
    def evaluate(self):
        self.model.eval()
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for img_batch, targets, _ in tqdm(self.test_loader):
                img_batch = img_batch.float().to(self.device)
                target_boxes = targets['bboxes'].float().to(self.device)[0]
                target_labels = targets['labels'].long().to(self.device)[0]
                
                _, detection_output = self.model(img_batch, None)
                
                pred_boxes = detection_output['boxes']
                pred_labels = detection_output['labels']
                pred_scores = detection_output['scores']
                
                # Format predictions
                pred_dict = {cls_name: [] for cls_name in self.dataset.class_to_index}
                gt_dict = {cls_name: [] for cls_name in self.dataset.class_to_index}
                
                for idx, box in enumerate(pred_boxes):
                    x1, y1, x2, y2 = box.detach().cpu().numpy()
                    label_idx = pred_labels[idx].detach().cpu().item()
                    score = pred_scores[idx].detach().cpu().item()
                    class_name = self.dataset.index_to_class[label_idx]
                    pred_dict[class_name].append([x1, y1, x2, y2, score])
                
                for idx, box in enumerate(target_boxes):
                    x1, y1, x2, y2 = box.detach().cpu().numpy()
                    label_idx = target_labels[idx].detach().cpu().item()
                    class_name = self.dataset.index_to_class[label_idx]
                    gt_dict[class_name].append([x1, y1, x2, y2])
                
                all_predictions.append(pred_dict)
                all_ground_truths.append(gt_dict)
        
        mean_ap, class_aps = calculate_mean_average_precision(
            all_predictions, all_ground_truths, aggregation_method='interp')
        
        print('\n=== Evaluation Results ===')
        print('Class-wise Average Precisions:')
        for idx in range(len(self.dataset.index_to_class)):
            class_name = self.dataset.index_to_class[idx]
            ap = class_aps[class_name]
            if not np.isnan(ap):
                print(f'  {class_name}: {ap:.4f}')
        
        print(f'\nMean Average Precision: {mean_ap:.4f}')
        
        return mean_ap, class_aps