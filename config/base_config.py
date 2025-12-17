"""Configuration loader and validator"""
import yaml

def load_configuration(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def validate_config(config):
    required_keys = ['dataset_params', 'model_params', 'train_params']
    for key in required_keys:
        assert key in config, f"Missing {key} in configuration"
    return True