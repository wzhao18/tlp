import json
import multiprocessing
import numpy as np
import glob
import pickle
import time
import torch
from torch import nn
from torch import optim
import os
from pathlib import Path
import time

from tlp_train import (GPTSegmentDataLoader, BertSegmentDataLoader, SegmentDataLoader,
                        LambdaRankLoss, validate)
from models import (AttentionModule, TransformerModule, TransformerEncoderLayerModule,
                    LSTMModule, GPTModule, BertModule)

from tvm import auto_scheduler
from tvm.tir.expr import FloatImm

def get_dataset_params(platform):
    with open(f"tlp_make_dataset_str_to_idx_{platform}.pkl", 'rb') as f:
        workloadkey_to_index, stepname_to_idx, auto_unroll_max_step_to_idx = pickle.load(f)
    
    stepname_to_idx_one_hot = {}
    for key, value in stepname_to_idx.items():
        one_hot = [0] * 11
        one_hot[stepname_to_idx[key]-1] = 1
        stepname_to_idx_one_hot[key] = one_hot

    if platform == 'llvm':
        max_seq_len = 54
        max_emb_size = 40
    else:
        max_seq_len = 69
        max_emb_size = 49

    if platform == 'llvm':
        crop_seq_len = 25
        crop_emb_size = 22
    else:
        crop_seq_len = 40
        crop_emb_size = 20

    env = (workloadkey_to_index, stepname_to_idx_one_hot,
           auto_unroll_max_step_to_idx, max_emb_size,
           max_seq_len, crop_emb_size, crop_seq_len)

    return env

def handle_file(file, workloadkey_to_index, stepname_to_idx_one_hot,
                auto_unroll_max_step_to_idx, max_emb_size, max_seq_len,
                crop_emb_size, crop_seq_len, use_bert):
    chw_dict = {
        'local': 1,
        'shared': 2,
        'global': 3,
    }

    with open(file, 'r') as f:
        lines = f.read().strip().split('\n')

    inputs, outputs = auto_scheduler.RecordReader(file).read_lines()
    task = auto_scheduler.measure.recover_measure_input(inputs[0]).task

    line_vecs = []

    min_cost = 1000000
    for line_idx, line in enumerate(lines):

        inp = json.loads(line)
        steps = inp['i'][1][1]

        step_vecs = []
        for st in steps:
            vec = []
            vec.extend(stepname_to_idx_one_hot[st[0]])

            for i, it in enumerate(st):
                if i == 0:
                    continue
                if isinstance(it, int):
                    vec.append(it)
                elif isinstance(it, list):
                    for ii in it:
                        assert isinstance(ii, int)
                        vec.append(ii)
                elif isinstance(it, str):
                    if st[0] == 'PR' and 'auto_unroll_max_step' in it:
                        vec.append(auto_unroll_max_step_to_idx[it])
                    elif st[0] == 'CHW':
                        vec.append(chw_dict[it])
                    elif st[0] == 'CHR' and it == 'shared':
                            vec.append(1)
                    else:
                        assert False
                else:
                    assert False

            assert len(vec) <= max_emb_size
            for i in range(len(vec), max_emb_size, 1):
                vec.append(0)

            vec = vec[:crop_emb_size]
            step_vecs.append(vec)

        if not use_bert:
            assert len(step_vecs) <= max_seq_len
            vec = [0] * crop_emb_size
            for i in range(len(step_vecs), max_seq_len, 1):
                step_vecs.append(vec.copy())
        step_vecs = step_vecs[:crop_seq_len]

        costs = [x.value for x in outputs[line_idx].costs if isinstance(x, FloatImm)]
        cost = np.mean(costs)
        line_vecs.append((step_vecs, cost))
        min_cost = min(min_cost, cost)
    line_vecs_new = []
    for line_vec in line_vecs:
        step_vecs, cost = line_vec
        score = min_cost / cost
        line_vecs_new.append((step_vecs, score, min_cost))
    line_vecs = line_vecs_new

    return task.workload_key, line_vecs


def to_token(file_vecs, step_vec_to_token_dict, crop_seq_len):
    print("to token ...")
    file_vecs_new = []
    for file_vec_idx, file_vec in enumerate(file_vecs):
        workload_key, line_vecs = file_vec

        line_vecs_new = []
        for line_vec in line_vecs:
            step_vecs, score, min_cost = line_vec

            step_vecs_new = [1]
            for st in step_vecs:
                step_vecs_new.append(step_vec_to_token_dict[str(st)] + 3)
            step_vecs_new.append(2)

            for i in range(len(step_vecs_new), crop_seq_len + 2, 1):
                step_vecs_new.append(3)

            line_vec = (step_vecs_new, score, min_cost)
            line_vecs_new.append(line_vec)

        file_vec = (workload_key, line_vecs_new)
        file_vecs_new.append(file_vec)

    return file_vecs_new


def aya_token(file_vecs):
    print("aya token ...")
    step_vec_to_token_set = set()

    for file_vec_idx, file_vec in enumerate(file_vecs):
        _, line_vecs = file_vec

        for line_vec in line_vecs:
            step_vecs, _, _ = line_vec

            for st in step_vecs:
                step_vec_to_token_set.add(str(st))

    step_vec_to_token_list = sorted(list(step_vec_to_token_set))
    step_vec_to_token_dict = {}
    for st_idx, st in enumerate(step_vec_to_token_list):
        step_vec_to_token_dict[str(st)] = st_idx + 1

    return step_vec_to_token_dict

def make_all_dataset(measurement_path, use_bert, workloadkey_to_index, stepname_to_idx_one_hot,
                        auto_unroll_max_step_to_idx, max_emb_size, max_seq_len,
                        crop_emb_size, crop_seq_len):
    json_files = sorted(glob.glob(f"{measurement_path}/*.json"))
    print(f"Number of measurement files: {len(json_files)}")

    multiprocessing_pool = multiprocessing.Pool()
    que_res_list = []
    for file in json_files:
        args = (file, workloadkey_to_index, stepname_to_idx_one_hot,
                auto_unroll_max_step_to_idx, max_emb_size,
                max_seq_len, crop_emb_size, crop_seq_len, use_bert)
        que_res_list.append(multiprocessing_pool.apply_async(handle_file, args=args))

    multiprocessing_pool.close()

    while True:
        count = 0
        for res in que_res_list:
            try:
                if res.successful():
                    count += 1
            except:
                pass
        if count == len(json_files):
            break
        else:
            print(f"Finished processing {count} files.")
            time.sleep(5)
        
    multiprocessing_pool.join()

    file_vecs = []
    for que_res in que_res_list:
        file_vecs.append(que_res.get())

    return file_vecs

def split_dataset(train_set_filename, test_set_filename, file_vecs, hold_out_tasks_set):
    train_and_val_dataset = []
    test_data = []

    for file_vec in file_vecs:

        workload_key, line_vecs = file_vec

        if workload_key in hold_out_tasks_set:
            test_data.append(file_vec)
        else:
            train_and_val_dataset.extend(line_vecs)
    
    print(f"Train and Val size: {len(train_and_val_dataset)} measurements")
    print(f"Test size: {len(test_data)} files")

    with open(test_set_filename, 'wb') as f:
        pickle.dump(test_data, f)
        print(f"Test data saved to {test_set_filename}")
    with open(train_set_filename, 'wb') as f:
        pickle.dump(train_and_val_dataset, f)
        print(f"Train and val data saved to {train_set_filename}")

def get_dataloader(train_dataset_filename, attention_class, train_size_per_gpu, val_size_per_gpu, n_gpu):

    print("Loading training dataset...")

    with open(train_dataset_filename, 'rb') as f:
        datasets_global = pickle.load(f)

    print("Training dataset is loaded.")

    datasets = np.array(datasets_global, dtype=object)
    train_len = int(len(datasets) * 0.9)
    perm = np.random.permutation(len(datasets))
    train_indices, val_indices = perm[:train_len], perm[train_len:]

    train_datas, val_datas = datasets[train_indices], datasets[val_indices]

    if attention_class == 'gpt':
        train_dataloader = GPTSegmentDataLoader(train_datas, train_size_per_gpu * n_gpu, True)
        val_dataloader = GPTSegmentDataLoader(val_datas, train_size_per_gpu * n_gpu, False)
    elif attention_class == 'bert':
        train_dataloader = BertSegmentDataLoader(train_datas, train_size_per_gpu * n_gpu, True)
        val_dataloader = BertSegmentDataLoader(val_datas, train_size_per_gpu * n_gpu, False)
    else:
        train_dataloader = SegmentDataLoader(train_datas, train_size_per_gpu * n_gpu, True)
        val_dataloader = SegmentDataLoader(val_datas, val_size_per_gpu * n_gpu, False)

    return train_dataloader, val_dataloader

def train(train_loader, val_dataloader, hardware, device, attention_class, rank_mse, n_epoch, optimizer,
            lr, weight_decay, save_dir, attention_head, step_size, fea_size, res_block_cnt,
            self_sup_model, n_gpus):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    
    
    model_save_file_name = f'{save_dir}/{attention_class}_cost_model_{hardware}.pkl'
    if os.path.exists(model_save_file_name):
        print(f"Model {model_save_file_name} exists. Skip training.")
        return model_save_file_name

    devices = list(range(n_gpus))

    # n_epoch = 50
    if attention_class == 'default':
        hidden_dim = [64, 128, 256, 256]
        out_dim = [256, 128, 64, 1]
        net = AttentionModule(fea_size, step_size, res_block_cnt,
                                hidden_dim, out_dim, attention_head).to(device)
        net = torch.nn.DataParallel(net, devices).to(torch.cuda.current_device())
    elif attention_class == 'transformer':
        print('TransformerModule')
        net = TransformerModule(fea_size, step_size, attention_head).to(device)
        net = torch.nn.DataParallel(net, devices).to(torch.cuda.current_device())
    elif attention_class == 'attention_encoder_layer':
        print('TransformerEncoderLayerModule')
        net = TransformerEncoderLayerModule(fea_size, step_size, attention_head).to(device)
        net = torch.nn.DataParallel(net, devices).to(torch.cuda.current_device())
    elif attention_class == 'lstm':
        print('LSTMModule')
        net = LSTMModule(fea_size, step_size).to(device)
        net = torch.nn.DataParallel(net, devices).to(torch.cuda.current_device())
    elif attention_class == 'gpt':
        print('GPTModule')
        net = GPTModule(self_sup_model).to(device)
        net = torch.nn.DataParallel(net, devices).to(torch.cuda.current_device())
    elif attention_class == 'bert':
        print('BertModule')
        net = BertModule(self_sup_model).to(device)
        net = torch.nn.DataParallel(net, devices).to(torch.cuda.current_device())
    elif attention_class == 'attention_512':
        hidden_dim = [64, 128, 256, 512]
        out_dim = [256, 128, 64, 1]
        print('Attention512Module')
        net = AttentionModule().to(device)
        net = torch.nn.DataParallel(net, devices).to(torch.cuda.current_device())
    elif attention_class == 'attention_768':
        hidden_dim = [64, 256, 512, 768]
        out_dim = [512, 256, 128, 1]
        print('Attention768Module')
        net = AttentionModule().to(device)
        net = torch.nn.DataParallel(net, devices).to(torch.cuda.current_device())
    elif attention_class == 'attention_1024':
        hidden_dim = [64, 256, 512, 1024]
        out_dim = [512, 256, 128, 1]
        print('Attention1024Module')
        net = AttentionModule().to(device)
        net = torch.nn.DataParallel(net, devices).to(torch.cuda.current_device())

    if rank_mse == 'rank':
        loss_func = LambdaRankLoss(device)
    else:
        loss_func = nn.MSELoss()

    n_epoch = n_epoch
    if optimizer == 'default':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=1)
    elif optimizer == 'decrease_per_17_0.8':
        print('optimizer', 'decrease_per_17_0.8')
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=0.8)
    elif optimizer == 'decrease_per_17_0.5':
        print('optimizer', 'decrease_per_17_0.5')
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=0.5)
    elif optimizer == 'decrease_per_12_0.5':
        print('optimizer', 'decrease_per_12_0.5')
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 4, gamma=0.5)
    elif optimizer == 'decrease_per_10_0.5':
        print('optimizer', 'decrease_per_10_0.5')
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 5, gamma=0.5)
    elif optimizer == 'decrease_per_17_0.5_no_decay':
        print('optimizer', 'decrease_per_17_0.5')
        optimizer = optim.Adam(net.parameters(), lr=lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=n_epoch // 3, gamma=0.5)

    lowest_val_loss = float('inf')
    train_loss = None
    print('start train...')
    print(len(train_loader), len(val_dataloader))
    for epoch in range(n_epoch):
        tic = time.time()

        net.train()
        train_loss = 0
        for batch, (batch_datas_steps, batch_labels) in enumerate(train_loader):
            batch_datas_steps = batch_datas_steps.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            loss = loss_func(net(batch_datas_steps), batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()
            train_loss += loss.item()
        lr_scheduler.step()

        train_time = time.time() - tic

        if epoch % 5 == 0 or epoch == n_epoch - 1 or True:

            valid_loss = validate(net, val_dataloader,
                                  loss_func, device=device)
            loss_msg = "Train Loss: %.4f\tValid Loss: %.4f" % (
                train_loss, valid_loss)
            print("Epoch: %d\tBatch: %d\t%s\tTrain Speed: %.0f" % (
                epoch, batch, loss_msg, len(train_loader) / train_time,))

            if valid_loss < lowest_val_loss:
                lowest_val_loss = valid_loss

                with open(model_save_file_name, 'wb') as f:
                    pickle.dump(net.cpu(), f)
                    print(f"Model is saved to {model_save_file_name}")
                net = net.to(device)
    
    return model_save_file_name

def pred_a_dataset(datas, task_pred_dict, model, device):

    datas_new = []
    for data_idx, data in enumerate([datas]):
        workloadkey, line_vecs = data
        datas_new.extend(line_vecs)

    if isinstance(model, BertModule):
        test_loader = BertSegmentDataLoader(datas_new, 512, False)
    elif isinstance(model, GPTModule):
        test_loader = GPTSegmentDataLoader(datas_new, 512, False)
    else:
        test_loader = SegmentDataLoader(datas_new, 4000, False)
    assert test_loader.min_latency.min() == test_loader.min_latency.max()

    preds_all = []
    labels_all = []

    total = 0
    for batch_datas_steps, batch_labels in test_loader:
        batch_datas_steps = batch_datas_steps.to(device)
        start = time.time()
        preds = model(batch_datas_steps)
        print(batch_datas_steps.shape)
        end = time.time()
        total += end - start
        # print(end - start)
        if isinstance(preds, list) and len(preds) > 1:
            preds = preds[0]
        preds_all.append(preds.detach().cpu())
        labels_all.append(batch_labels.detach().cpu())

    preds_all = torch.cat(preds_all, dim=0)
    labels_all = torch.cat(labels_all, dim=0)
    task_pred_dict[workloadkey] = (preds_all.detach().cpu().numpy(
    ), test_loader.min_latency.min().numpy(), labels_all.numpy())
    return total

def eval_model(test_datasets, model_file, dataset_path, device, platform):

    top_ks = [1, 5, 10, 20]

    with open(model_file, 'rb') as f:
        model = pickle.load(f).module.to(device)
    model.eval()
    task_pred_dict = {}

    pred_a_dataset_dict = {}
    for data_idx, data in enumerate(test_datasets):
        workloadkey, line_vecs = data
        pred_a_dataset_dict[workloadkey] = data

    files = []

    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for layer in [18, 50]:
                files.append(f'network_info/((resnet_{layer},[({batch_size},3,{image_size},{image_size})]),{platform}).task.pkl')
    for batch_size in [1, 4, 8]:
        for image_size in [224, 240, 256]:
            for name in ['mobilenet_v2', 'mobilenet_v3']:
                files.append(f'network_info/(({name},[({batch_size},3,{image_size},{image_size})]),{platform}).task.pkl')
    for batch_size in [1, 2, 4]:
        for seq_length in [64, 128, 256]:
            for scale in ['tiny', 'base']:
                files.append(f'network_info/((bert_{scale},[({batch_size},{seq_length})]),{platform}).task.pkl')
    for batch_size in [1, 2, 4]:
            for image_size in [299]:
                files.append(f'network_info/((inception_v3,[({batch_size},3,{image_size},{image_size})]),{platform}).task.pkl')

    top_1_total = []
    top_5_total = []
    top_10_total = []
    top_20_total = []
    best_latency_total_list = []
    best_latency_total = 0
    top1_total = 0
    top5_total = 0
    top10_total = 0
    top20_total = 0
    for file in files:
        path = Path(dataset_path)
        tasks, task_weights = pickle.load(open(path / file, "rb"))
        latencies = [0] * len(top_ks)
        best_latency = 0

        time = 0
        for task, weight in zip(tasks, task_weights):
            if task.workload_key not in pred_a_dataset_dict:
                print('error task.workload_key not in pred_a_dataset_dict')
                continue
            time += pred_a_dataset(
                pred_a_dataset_dict[task.workload_key], task_pred_dict, model, device)
            preds, min_latency, labels = task_pred_dict[task.workload_key]

            real_values = labels[np.argsort(-preds)]
            real_latency = min_latency / np.maximum(real_values, 1e-5)

            for i, top_k in enumerate(top_ks):
                latencies[i] += np.min(real_latency[:top_k]) * weight
            best_latency += min_latency * weight

        top_1_total.append(best_latency/latencies[0])
        print(f"top 1 score: {best_latency/latencies[0]}")
        top_5_total.append(best_latency / latencies[1])
        print(f"top 5 score: {best_latency / latencies[1]}")

        best_latency_total_list.append(best_latency)
        best_latency_total += best_latency
        top1_total += latencies[0]
        top5_total += latencies[1]
        top10_total += latencies[2]
        top20_total += latencies[3]

    print("???")
    print(time)
    print(f"average top 1 score is {best_latency_total / top1_total}")
    top_1_total.append(best_latency_total / top1_total)
    print(f"average top 5 score is {best_latency_total / top5_total}")
    top_5_total.append(best_latency_total / top5_total)
    print(f"average top 10 score is {best_latency_total / top10_total}")
    top_10_total.append(best_latency_total / top1_total)
    print(f"average top 20 score is {best_latency_total / top20_total}")
    top_20_total.append(best_latency_total / top5_total)