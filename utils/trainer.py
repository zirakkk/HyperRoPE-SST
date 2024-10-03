import torch
import torch.nn as nn
import torch.optim as optim
from models.hyperrope_vit import HyperRopeViT
from utils.utils import device, recorder
from utils.evaluation import HSIEvaluation
import numpy as np
from typing import Dict, Tuple
from configs.config import CHECKPOINT_PATH_PREFIX

class BaseTrainer:
    def __init__(self, params: Dict):
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.device = device
        self.evaluator = HSIEvaluation(param=params)

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.clip = self.train_params.get('clip', 15)

       
    def train(self, train_loader, valid_loader=None):
        torch.autograd.set_detect_anomaly(True)
        epochs = self.params['train'].get('epochs', 200)
        best_valid_oa = 0
        patience = self.params['train'].get('patience', 10)
        no_improve_count = 0

        for epoch in range(epochs):
            self.net.train()
            epoch_loss = 0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.net(data)
                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            recorder.append_index_value("epoch_loss", epoch + 1, avg_loss)
            print(f'[Epoch: {epoch + 1}] [Loss: {avg_loss:.5f}]')

            if valid_loader is not None and (epoch+1) % 1 == 0:
                valid_oa = self.validate(valid_loader, epoch)
                if valid_oa > best_valid_oa:
                    best_valid_oa = valid_oa
                    self.save_checkpoint(epoch, best_valid_oa, is_best=True)  # Save best model
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= patience:
                        print(f'Early stopping after {epoch + 1} epochs')
                        break
                    
        self.save_checkpoint(epoch, best_valid_oa, is_best=False)  # Save last model
        print('Finished Training')
        return True

    def validate(self, valid_loader, epoch):
        self.net.eval()
        y_pred_valid, y_valid = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(self.device)
                outputs = self.net(inputs)
                outputs = outputs.detach().cpu().numpy()
                y_pred_valid.append(np.argmax(outputs, axis=1))
                y_valid.append(labels.numpy())
        
        y_pred_valid = np.concatenate(y_pred_valid)
        y_valid = np.concatenate(y_valid)
        temp_res = self.evaluator.eval(y_valid, y_pred_valid)
        
        recorder.append_index_value("valid_oa", epoch+1, temp_res['oa'])
        recorder.append_index_value("valid_aa", epoch+1, temp_res['aa'])
        recorder.append_index_value("valid_kappa", epoch+1, temp_res['kappa'])
        print(f'[Validation] [Epoch: {epoch+1}] [OA: {temp_res["oa"]:.5f}] [AA: {temp_res["aa"]:.5f}] [Kappa: {temp_res["kappa"]:.5f}]')
        
        return temp_res['oa']

    def save_checkpoint(self, epoch, valid_oa, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'valid_oa': valid_oa,
        }
        
        if is_best:
            torch.save(checkpoint, f"{CHECKPOINT_PATH_PREFIX}/{self.params['data']['data_sign']}_best_model.pth")
            print(f'Best model checkpoint saved at epoch {epoch}')
        else:
            torch.save(checkpoint, f"{CHECKPOINT_PATH_PREFIX}/{self.params['data']['data_sign']}_last_model.pth")
            print(f'Last model checkpoint saved at epoch {epoch}')


    def final_eval(self, test_loader):
        checkpoint = torch.load(f"{CHECKPOINT_PATH_PREFIX}/{self.params['data']['data_sign']}_best_model.pth")
        self.net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded from epoch {checkpoint['epoch']} with OA {checkpoint['valid_oa']}")
        y_pred_test, y_test = self.test(test_loader)
        return self.evaluator.eval(y_test, y_pred_test)


    def test(self, test_loader) -> Tuple[np.ndarray, np.ndarray]:
        self.net.eval()
        y_pred_test = []
        y_test = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.net(inputs)
                outputs = outputs.detach().cpu().numpy()
                y_pred_test.append(np.argmax(outputs, axis=1))
                y_test.append(labels.numpy())
        
        return np.concatenate(y_pred_test), np.concatenate(y_test)

class HyperRopeViTTrainer(BaseTrainer):
    def __init__(self, params: Dict):
        super(HyperRopeViTTrainer, self).__init__(params)
        
        self.net = HyperRopeViT(self.params).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=5, verbose=True)
    
    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)
    
    def train(self, train_loader, valid_loader=None):
        result = super().train(train_loader, valid_loader)
#        if valid_loader:
#           self.scheduler.step(self.best_valid_oa)
        return result

def get_trainer(params: Dict):
    trainer_type = params['net']['trainer']
    if trainer_type == "hyperrope_vit":
        return HyperRopeViTTrainer(params)
    raise Exception("Trainer not implemented!")