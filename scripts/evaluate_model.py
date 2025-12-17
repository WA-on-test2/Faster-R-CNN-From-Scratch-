import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from config.base_config import load_configuration
from core.models.detector import ObjectDetectionModel
from data.loaders.voc_loader import VOCDatasetLoader
from controllers.evaluator import ModelEvaluator
from utils.common import get_computation_device

def main():
    parser = argparse.ArgumentParser(description='Evaluate object detection model')
    parser.add_argument('--config', type=str, default='config/voc_dataset.yaml')
    args = parser.parse_args()
    
    config = load_configuration(args.config)
    device = get_computation_device()
    
    test_dataset = VOCDatasetLoader(
        'test',
        config['dataset_params']['im_test_path'],
        config['dataset_params']['ann_test_path'])
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = ObjectDetectionModel(
        config['model_params'],
        total_classes=config['dataset_params']['num_classes'])
    
    checkpoint_path = f"{config['train_params']['task_name']}/{config['train_params']['ckpt_name']}"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    model.to(device)
    evaluator = ModelEvaluator(model, test_loader, test_dataset)
    evaluator.evaluate()

if __name__ == '__main__':
    main()