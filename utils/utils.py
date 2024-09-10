import json
import datetime
import torch
from typing import Dict, Any

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config_path_prefix = './configs'

class AvgrageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

    def get_avg(self) -> float:
        return self.avg

    def get_num(self) -> int:
        return self.cnt
    
class HSIRecoder:
    def __init__(self):
        self.record_data: Dict[str, Any] = {}
        self.pred = None

    def append_index_value(self, name: str, index: int, value: Any):
        if name not in self.record_data:
            self.record_data[name] = {
                "type": "index_value",
                "index": [],
                "value": []
            } 
        self.record_data[name]['index'].append(index)
        self.record_data[name]['value'].append(value)
    
    def record_time(self, time: float):
        self.record_data['eval_time'] = time

    def record_param(self, param: Dict):
        self.record_data['param'] = param 

    def record_eval(self, eval_obj: Dict):
        self.record_data['eval'] = eval_obj
        
    def record_pred(self, pred_matrix):
        self.pred = pred_matrix

    def to_file(self, path: str):
        time_stamp = datetime.datetime.now().strftime('%m_%d_%Y_%I%p')
        save_path_json = f"{path}_{time_stamp}.json"
        #save_path_pred = f"{path}_{time_stamp}.pred.npy"

        with open(save_path_json, 'w') as fout:
            json.dump(self.record_data, fout, indent=4)

        print(f"Save record of {path} done!")
        
    def reset(self):
        self.record_data = {}

# global recorder
recorder = HSIRecoder()