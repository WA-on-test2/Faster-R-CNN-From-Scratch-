import torch
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common import get_computation_device, set_random_seed

class ModelTrainer:
    def __init__(self, model, train_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = get_computation_device()
        self.optimizer = torch.optim.SGD(
            lr=config['train_params']['lr'],
            params=filter(lambda p: p.requires_grad, model.parameters()),
            weight_decay=5e-4,
            momentum=0.9)
        
        self.scheduler = MultiStepLR(
            self.optimizer, 
            milestones=config['train_params']['lr_steps'],
            gamma=0.1)
        
        self.accumulation_steps = config['train_params']['acc_steps']
        
    def train_epoch(self):
        self.model.train()
        
        rpn_cls_losses, rpn_loc_losses = [], []
        roi_cls_losses, roi_loc_losses = [], []   
        self.optimizer.zero_grad()
        step_counter = 1
        
        for img_batch, target_batch, _ in tqdm(self.train_loader):
            img_batch = img_batch.float().to(self.device)
            target_batch['bboxes'] = target_batch['bboxes'].float().to(self.device)
            target_batch['labels'] = target_batch['labels'].long().to(self.device)
            
            rpn_out, roi_out = self.model(img_batch, target_batch)
            
            rpn_loss = rpn_out['rpn_classification_loss'] + rpn_out['rpn_localization_loss']
            roi_loss = roi_out['frcnn_classification_loss'] + roi_out['frcnn_localization_loss']
            total_loss = rpn_loss + roi_loss
            
            rpn_cls_losses.append(rpn_out['rpn_classification_loss'].item())
            rpn_loc_losses.append(rpn_out['rpn_localization_loss'].item())
            roi_cls_losses.append(roi_out['frcnn_classification_loss'].item())
            roi_loc_losses.append(roi_out['frcnn_localization_loss'].item())
            
            total_loss = total_loss / self.accumulation_steps
            total_loss.backward()
            
            if step_counter % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            step_counter += 1
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return {
            'rpn_cls': rpn_cls_losses,
            'rpn_loc': rpn_loc_losses,
            'roi_cls': roi_cls_losses,
            'roi_loc': roi_loc_losses
        }
    
    def train(self, num_epochs):
        """Full training loop"""
        output_dir = self.config['train_params']['task_name']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for epoch_idx in range(num_epochs):
            losses = self.train_epoch()
            
            print(f'Epoch {epoch_idx} completed')
            print(f"RPN Cls: {np.mean(losses['rpn_cls']):.4f} | "
                  f"RPN Loc: {np.mean(losses['rpn_loc']):.4f} | "
                  f"ROI Cls: {np.mean(losses['roi_cls']):.4f} | "
                  f"ROI Loc: {np.mean(losses['roi_loc']):.4f}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, 
                                          self.config['train_params']['ckpt_name'])
            torch.save(self.model.state_dict(), checkpoint_path)
            
            self.scheduler.step()