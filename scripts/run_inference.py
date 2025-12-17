import argparse
import torch
import cv2
import random
import os
from tqdm import tqdm

from config.base_config import load_configuration
from core.models.detector import ObjectDetectionModel
from data.loaders.voc_loader import VOCDatasetLoader
from utils.visualization import draw_bounding_boxes
from utils.common import get_computation_device

def main():
    parser = argparse.ArgumentParser(description='Run inference visualization')
    parser.add_argument('--config', type=str, default='config/voc_dataset.yaml')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='inference_samples')
    args = parser.parse_args()
    
    config = load_configuration(args.config)
    device = get_computation_device()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    test_dataset = VOCDatasetLoader(
        'test',
        config['dataset_params']['im_test_path'],
        config['dataset_params']['ann_test_path'])
    
    model = ObjectDetectionModel(
        config['model_params'],
        total_classes=config['dataset_params']['num_classes'])
    
    checkpoint_path = f"{config['train_params']['task_name']}/{config['train_params']['ckpt_name']}"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    model.classification_head.min_confidence_threshold = 0.7
    for sample_idx in tqdm(range(args.num_samples)):
        random_idx = random.randint(0, len(test_dataset) - 1)
        img_tensor, targets, img_path = test_dataset[random_idx]
        gt_image = cv2.imread(img_path)
        gt_boxes = targets['bboxes'].detach().cpu().numpy()
        gt_labels = targets['labels'].detach().cpu().numpy()
        
        gt_result = draw_bounding_boxes(
            gt_image, gt_boxes, gt_labels, None,
            test_dataset.index_to_class, color=(0, 255, 0))
        
        cv2.imwrite(f'{args.output_dir}/ground_truth_{sample_idx}.png', gt_result)
        
        with torch.no_grad():#visualiz.
            img_input = img_tensor.unsqueeze(0).float().to(device)
            _, detection_output = model(img_input, None)
        
        pred_boxes = detection_output['boxes'].detach().cpu().numpy()
        pred_labels = detection_output['labels'].detach().cpu().numpy()
        pred_scores = detection_output['scores'].detach().cpu().numpy()
        
        pred_image = cv2.imread(img_path)
        pred_result = draw_bounding_boxes(
            pred_image, pred_boxes, pred_labels, pred_scores,
            test_dataset.index_to_class, color=(0, 0, 255))
        
        cv2.imwrite(f'{args.output_dir}/prediction_{sample_idx}.jpg', pred_result)
    
    print(f'Generated {args.num_samples} visualizations in {args.output_dir}/')

if __name__ == '__main__':
    main()