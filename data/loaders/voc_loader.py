import glob
import os
import random
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


def parse_dataset_annotations(image_directory, annotation_directory, class_mapping):
    dataset_entries = []
    
    annotation_files = glob.glob(os.path.join(annotation_directory, '*.xml'))
    
    for annotation_path in tqdm(annotation_files):
        entry_info = {}
        image_id = os.path.basename(annotation_path).split('.xml')[0]
        entry_info['img_id'] = image_id
        entry_info['filename'] = os.path.join(image_directory, f'{image_id}.jpg')

        xml_tree = ET.parse(annotation_path)
        xml_root = xml_tree.getroot()
        
        size_element = xml_root.find('size')
        img_width = int(size_element.find('width').text)
        img_height = int(size_element.find('height').text)
        entry_info['width'] = img_width
        entry_info['height'] = img_height
        
        object_annotations = []
        
        for obj_element in xml_tree.findall('object'):
            annotation_dict = {}
            
            class_name = obj_element.find('name').text
            class_index = class_mapping[class_name]
            
            bbox_element = obj_element.find('bndbox')
            bbox_coords = [
                int(float(bbox_element.find('xmin').text)) - 1,
                int(float(bbox_element.find('ymin').text)) - 1,
                int(float(bbox_element.find('xmax').text)) - 1,
                int(float(bbox_element.find('ymax').text)) - 1
            ]
            
            annotation_dict['label'] = class_index
            annotation_dict['bbox'] = bbox_coords
            object_annotations.append(annotation_dict)
        
        entry_info['detections'] = object_annotations
        dataset_entries.append(entry_info)
    
    print(f'Total {len(dataset_entries)} images found')
    return dataset_entries


class VOCDatasetLoader(Dataset):
    def __init__(self, data_split, image_directory, annotation_directory):
        self.data_split = data_split
        self.image_dir = image_directory
        self.annotation_dir = annotation_directory
        
        class_names = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        
        class_names = sorted(class_names)
        class_names = ['background'] + class_names
        
        self.class_to_index = {class_names[i]: i for i in range(len(class_names))}
        self.index_to_class = {i: class_names[i] for i in range(len(class_names))}
        
        print(self.index_to_class)
        
        self.dataset_entries = parse_dataset_annotations(
            image_directory, annotation_directory, self.class_to_index)
    
    def __len__(self):
        return len(self.dataset_entries)
    
    def __getitem__(self, index):
        entry_data = self.dataset_entries[index]
        image_obj = Image.open(entry_data['filename'])
        should_flip = False
        if self.data_split == 'train' and random.random() < 0.5:
            should_flip = True
            image_obj = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
        
        image_tensor = torchvision.transforms.ToTensor()(image_obj)
        
        target_dict = {}
        target_dict['bboxes'] = torch.as_tensor(
            [det['bbox'] for det in entry_data['detections']])
        target_dict['labels'] = torch.as_tensor(
            [det['label'] for det in entry_data['detections']])
        
        if should_flip:
            for box_idx, box in enumerate(target_dict['bboxes']):
                x1, y1, x2, y2 = box
                box_width = x2 - x1
                image_width = image_tensor.shape[-1]
                
                new_x1 = image_width - x1 - box_width
                new_x2 = new_x1 + box_width
                
                target_dict['bboxes'][box_idx] = torch.as_tensor([new_x1, y1, new_x2, y2])
        
        return image_tensor, target_dict, entry_data['filename']