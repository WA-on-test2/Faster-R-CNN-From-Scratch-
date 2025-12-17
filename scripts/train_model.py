import argparse
import torch
from torch.utils.data import DataLoader

from config.base_config import load_configuration
from core.models.detector import ObjectDetectionModel
from data.loaders.voc_loader import VOCDatasetLoader
from controllers.trainer import ModelTrainer
from utils.common import set_random_seed, get_computation_device

def main():
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--config', type=str, default='config/voc_dataset.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    config = load_configuration(args.config)
    set_random_seed(config['train_params']['seed'])
    device = get_computation_device()
    train_dataset = VOCDatasetLoader(
        'train',
        config['dataset_params']['im_train_path'],
        config['dataset_params']['ann_train_path'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4)

    model = ObjectDetectionModel(
        config['model_params'],
        num_classes=config['dataset_params']['num_classes'])
    model.train()
    model.to(device)
    
    trainer = ModelTrainer(model, train_loader, config)
    trainer.train(config['train_params']['num_epochs'])
    
    print('Training completed!')

if __name__ == '__main__':
    main()