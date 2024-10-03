import os, time, json, argparse
from typing import Dict
from utils.utils import recorder
from data.data_loader import MultiFileHSIDataLoader
from utils.trainer import get_trainer
from configs.config import DEFAULT_RES_SAVE_PATH_PREFIX
from configs.config import CHECKPOINT_PATH_PREFIX
from configs.config import CONFIG_PATH_PREFIX


def train_by_param(param: Dict):
    recorder.reset()
    dataloader = MultiFileHSIDataLoader(param)
    train_loader, valid_loader, test_loader = dataloader.generate_torch_dataset() 
    trainer = get_trainer(param)
    start_train_time = time.time()
    trainer.train(train_loader, valid_loader)
    end_train_time = time.time()
    train_time = end_train_time - start_train_time
    print(f"Train time: {train_time:.2f} seconds") 
    recorder.record_time(train_time)
    start_eval_time = time.time()
    eval_res = trainer.final_eval(test_loader)
    end_eval_time = time.time()
    eval_time = end_eval_time - start_eval_time
    print(f"Eval time: {eval_time:.2f} seconds") 
    recorder.record_time(eval_time)
    recorder.record_param(param)
    recorder.record_eval(eval_res)
    return recorder

def run_experiment(config_file: str, dataset_name: str):
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    os.makedirs(save_path_prefix, exist_ok=True)
    os.makedirs(CHECKPOINT_PATH_PREFIX, exist_ok=True)
    
    path_param = os.path.join(CONFIG_PATH_PREFIX, config_file)
    with open(path_param, 'r') as fin:
        org_params = json.load(fin)

    # Combine common parameters with dataset-specific parameters from json file
    param = {
        'data': {
            **org_params['datasets'][dataset_name],
            **org_params['common']   
        },
        'net': org_params['net'],
        'train': org_params['train']
    }

    # Set unique name for the experiment
    uniq_name = f"Hyper2DRoPE_Classification_Model"
    
    data_sign = param['data']['data_sign']
    print(f'...Starting training for {uniq_name} on {data_sign} dataset...')
    train_by_param(param)
    print(f'...Model evaluation completed for {uniq_name} on {data_sign} dataset')
    path = os.path.join(save_path_prefix, f"{data_sign}_{uniq_name}")
    recorder.to_file(path)

def main():
    parser = argparse.ArgumentParser(description="Run HSI classification experiment")
    parser.add_argument("--config", type=str, default="hyper2Drope.json", help="Configuration file name")
    parser.add_argument("--dataset", type=str, default="Plastic", choices=["Plastic", "IndianPine", "Houston", "Pavia"], help="Dataset to use")
    args = parser.parse_args()

    run_experiment(args.config, args.dataset)

if __name__ == "__main__":
    main()