import pickle
import os

from common import load_and_register_tasks
from exp_util import (make_all_dataset, split_dataset, get_dataset_params,
                        get_dataloader, eval_model, train, aya_token, to_token)
from tlp_train import set_seed

# Stages

# Parameters - will set them manually here

holdout_models = []

for batch_size in [1, 4, 8]:
    for image_size in [224, 240, 256]:
        for layer in [18, 50]:
            holdout_models.append(f'(resnet_{layer},[({batch_size},3,{image_size},{image_size})])')
for batch_size in [1, 4, 8]:
    for image_size in [224, 240, 256]:
        for name in ['mobilenet_v2', 'mobilenet_v3']:
            holdout_models.append(f'({name},[({batch_size},3,{image_size},{image_size})])')
for batch_size in [1, 2, 4]:
    for seq_length in [64, 128, 256]:
        for scale in ['tiny', 'base']:
            holdout_models.append(f'(bert_{scale},[({batch_size},{seq_length})])')
for batch_size in [1, 2, 4]:
        for image_size in [299]:
            holdout_models.append(f'(inception_v3,[({batch_size},3,{image_size},{image_size})])')

hardware_platform = {
    "e5-2673": "llvm",
    # "epyc-7452": "llvm",
    # "graviton2": "llvm",
    # "i7": "llvm",
    # "platinum-8272": "llvm",
    # "k80": "cuda",
    # "t4": "cuda"
}

train_size_per_gpu = 1024
val_size_per_gpu = 1024
n_gpu = 4

device = 'cuda:0'
attention_class = 'bert'
rank_mse = 'rank'
n_epoch = 50
optimizer = 'default'
lr = 7e-4
weight_decay = 1e-6
attention_head = 8
step_size = 25
fea_size = 22
res_block_cnt = 2
self_sup_model = ''
use_bert = attention_class == 'bert'

dataset_suffix = ""
if use_bert:
    dataset_suffix = '_bert'

def main():
    set_seed(0)

    for hardware, platform in hardware_platform.items():
        dataset_path = "/home/zhaowe58/tenset_dataset/"
        if platform == "llvm":
            dataset_path += "dataset_cpu"
        else:
            dataset_path += "dataset_gpu"
        measurement_path = f"{dataset_path}/measure_records/{hardware}"

        print(f"Running hardware: {hardware}")

        # Load all tasks
        load_and_register_tasks(dataset_path)

        # ======================= Dataset Preprocessing ===========================

        # Hold out tasks
        train_set_filename = f"tlp_train_and_val_{hardware}{dataset_suffix}.pkl"
        test_set_filename = f"tlp_test_{hardware}{dataset_suffix}.pkl"
        if not os.path.exists(train_set_filename) or not os.path.exists(test_set_filename):
            hold_out_tasks = []
            for model in holdout_models:
                file_path = f"{dataset_path}/network_info/({model},{platform}).task.pkl"
                tasks, _ = pickle.load(open(file_path, "rb"))
                hold_out_tasks.extend(tasks)
            hold_out_task_keys = set([task.workload_key for task in hold_out_tasks])

            dataset_params = get_dataset_params(platform)
            file_vecs = make_all_dataset(measurement_path, use_bert, *dataset_params)
            
            if use_bert:
                crop_seq_len = dataset_params[6]
                step_vec_to_token_dict = aya_token(file_vecs)
                file_vecs = to_token(file_vecs, step_vec_to_token_dict, crop_seq_len)
            
            split_dataset(train_set_filename, test_set_filename, file_vecs, hold_out_task_keys)

        # ===================== End Dataset Preprocessing ==========================

        # ============================= Training ===================================

        train_dataloader, val_dataloader = get_dataloader(train_set_filename, attention_class,
                                                            train_size_per_gpu, val_size_per_gpu, n_gpu)
        
        model_filename = train(train_dataloader, val_dataloader, hardware, device, attention_class, rank_mse,
                                n_epoch, optimizer, lr, weight_decay, "./models", attention_head, step_size,
                                fea_size, res_block_cnt, self_sup_model, n_gpu)

        # =========================== End Training =================================

        # ============================ Evaluation ==================================
        
        with open(test_set_filename, 'rb') as f:
            test_datasets = pickle.load(f)

        eval_model(test_datasets, model_filename, dataset_path, device, platform)

        # ========================== End Evaluation ================================

if __name__ == "__main__":
    main()