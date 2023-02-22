import wandb

import torch
from torch import Tensor 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from better_mistakes.model.losses import HierarchicalCrossEntropyLoss
from better_mistakes.trees import get_weighting
from better_mistakes.model.labels import make_all_soft_labels, make_batch_soft_labels

from datetime import datetime
import argparse
import os
import json
from typing import *

from model import init_model
from data import make_kshot_loader, make_dataloader
from loss import CPCCLoss, QuadrupletLoss, GroupLasso
from param import init_optim_schedule, load_params
from utils import get_layer_prob_from_fine, seed_everything


def pretrain_objective(train_loader : DataLoader, test_loader : DataLoader, device : torch.device, 
               save_dir : str, seed : int, split : str, CPCC : bool, exp_name : str, epochs : int,
               task : str, dataset_name : str, breeds_setting : str, hyper) -> None:
    '''
    Pretrain session wrapper. Use extra epochs for curriculum learning.
    Args:
        train_loader : dataset train loader.
        test_loader : dataset test loader
        device : cpu/gpu
        save_dir : directory to save model checkpoint
        seed : random state
        split : split/full
        CPCC : use CPCC as a regularizer or not
        exp_name : experiment name
        epochs : number of training iterations for all experiments except for curriculum
        dataset_name : MNIST/CIFAR/BREEDS
    '''
    def one_stage_pretrain(train_loader : DataLoader, test_loader : DataLoader, 
                     device : torch.device, save_dir : str, seed : int, split : str, 
                     CPCC : bool, exp_name : str, epochs : int, task : str='') -> nn.Module:
        '''
        Main train/test loop for all experiments except for curriculum learning.
        Args:
            train_loader : dataset train loader.
            test_loader : dataset test loader
            device : cpu/gpu
            save_dir : directory to save model checkpoint
            seed : random state
            split : split/full
            CPCC : use CPCC as a regularizer or not
            exp_name : experimental setting, can't be curriculum
            epochs : number of training iterations
            task : in/sub
        '''
        # ================= SETUP STARTS =======================
        assert 'curriculum' not in exp_name, 'Invalid experiment'
        if task == 'in':
            dataset = train_loader.dataset.dataset
        else:
            dataset = train_loader.dataset
        classes = dataset.fine_names
        if 'MTL' in exp_name:
            num_classes = [len(dataset.coarse_names), len(dataset.fine_names)]
        else:
            num_classes = [len(dataset.fine_names)]
        coarse_targets_map = dataset.coarse_map
        num_train_batches = len(train_loader.dataset) // train_loader.batch_size + 1
        last_train_batch_size = len(train_loader.dataset)  % train_loader.batch_size
        num_test_batches = len(test_loader.dataset) // test_loader.batch_size + 1
        last_test_batch_size = len(test_loader.dataset)  % test_loader.batch_size

        if CPCC:
            exp_name = exp_name + 'CPCC'

        if group:
            exp_name = exp_name + 'Group'

        split = split + task

        optim_config, scheduler_config = hyper['optimizer'], hyper['scheduler']
        init_config = {"dataset":dataset_name,
                    "exp_name":exp_name,
                    "_batch_size":train_loader.batch_size,
                    "epochs":epochs,
                    "_optimizer":'SGD',
                    "_scheduler":'StepLR',
                    "seed":seed, # will be replaced until we reach the max seeds
                    "save_dir":save_dir,
                    "_num_workers":train_loader.num_workers,
                    "split":split,
                    "lamb":lamb,
                    "breeds_setting":breeds_setting,
                    }
        config = {**init_config, **optim_config, **scheduler_config}        

        torch.autograd.set_detect_anomaly(True)
        if 'HXE' in exp_name:
            hierarchy = dataset.build_nltk_tree()
            alpha = 0.4
            config['alpha'] = alpha
            weights = get_weighting(hierarchy, "exponential", value=alpha, normalize=False)
            criterion_ce = HierarchicalCrossEntropyLoss(hierarchy, classes, weights).to(device)
        elif 'soft' in exp_name:
            beta = 10
            config['beta'] = beta
            distances = dataset.make_distances()
            all_soft_labels = make_all_soft_labels(distances, list(range(num_classes[0])), beta)
            criterion_ce = nn.KLDivLoss(reduction='batchmean').to(device) # "entropy flavor", soft version
        elif 'quad' in exp_name:
            m1 = 0.25
            m2 = 0.15
            lambda_s = 0.8
            config['m1'] = m1
            config['m2'] = m2
            config['lambda_s'] = lambda_s
            criterion_quad = QuadrupletLoss(train_loader.dataset, m1, m2).to(device)
            criterion_ce = nn.CrossEntropyLoss().to(device)
        else:
            criterion_ce = nn.CrossEntropyLoss().to(device)
        criterion_cpcc = CPCCLoss(dataset, cpcc_layers, cpcc_metric).to(device)
        criterion_group = GroupLasso(dataset).to(device)
        
        with open(save_dir+'/config.json', 'w') as fp:
            json.dump(config, fp, sort_keys=True, indent=4)
        
        out_dir = save_dir+f"/{split}_seed{seed}.pth"

        if os.path.exists(out_dir):
            print("Skipped.")
            return
            # msg = input("File Exists. Override the model? (y/n)")
            # if msg.lower() == 'n':
            #     print("Skipped.")
            #     return
            # elif msg.lower() == 'y':
            #     print("Retrain model.")
        
        wandb.init(project=f"{dataset_name}_onestage_pretrain", 
                entity="structured_task",
                name=datetime.now().strftime("%m%d%Y%H%M%S"),
                config=config)

        model = init_model(dataset_name, num_classes, device)
        optimizer, scheduler = init_optim_schedule(model, hyper)
        
        # ================= SETUP ENDS =======================

        def get_different_loss(exp_name : str, model : nn.Module, data : Tensor, 
                        criterion_ce : nn.Module, target_fine : Tensor, target_coarse : Tensor, 
                        coarse_targets_map : list, idx : int, num_batches : int, 
                        last_batch_size : int, num_classes : int, 
                        all_soft_labels=None, lambda_s : float = None, criterion_quad : nn.Module = None) -> Tuple[Tensor, Tensor, Tensor]:
            '''
                Helper to calculate non CPCC loss, also return representation and model raw output
            '''
            if 'MTL' in exp_name:
                representation, output_coarse, output_fine = model(data)
                loss_ce_coarse = criterion_ce(output_fine, target_fine)
                loss_ce_fine = criterion_ce(output_coarse, target_coarse)
                loss_ce = loss_ce_coarse + loss_ce_fine
            else:
                representation, output_fine = model(data)
                if 'sumloss' in exp_name:
                    loss_ce_fine = criterion_ce(output_fine, target_fine)
                    prob_fine = F.softmax(output_fine,dim=1)
                    prob_coarse = get_layer_prob_from_fine(prob_fine, coarse_targets_map)
                    loss_ce_coarse = F.nll_loss(torch.log(prob_coarse), target_coarse)
                    loss_ce = loss_ce_fine + loss_ce_coarse
                elif 'soft' in exp_name:
                    if idx == num_batches - 1:
                        target_distribution = make_batch_soft_labels(all_soft_labels, target_fine, num_classes, last_batch_size, device)
                    else:
                        target_distribution = make_batch_soft_labels(all_soft_labels, target_fine, num_classes, batch_size, device)
                    prob_fine = F.softmax(output_fine,dim=1)
                    loss_ce = criterion_ce(prob_fine.log(), target_distribution)
                elif 'quad' in exp_name:
                    loss_quad = criterion_quad(F.normalize(representation, dim=-1), target_fine)
                    loss_cee = criterion_ce(output_fine, target_fine)
                    loss_ce = (1 - lambda_s) * loss_quad + lambda_s * loss_cee
                else:
                    loss_ce = criterion_ce(output_fine, target_fine)
            return representation, output_fine, loss_ce

        for epoch in range(epochs):
            t_start = datetime.now() # record the time for each epoch
            model.train()
            train_fine_accs = []
            train_coarse_accs = []
            train_losses_ce = []
            train_losses_cpcc = []
            train_losses_group = []
            
            for idx, (data, _, target_coarse, _, target_fine)  in enumerate(train_loader):
                data = data.to(device)
                target_fine = target_fine.to(device)
                target_coarse = target_coarse.to(device)

                optimizer.zero_grad()
                
                if 'soft' in exp_name:
                    representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                                            coarse_targets_map, idx, num_train_batches, 
                                            last_train_batch_size, num_classes[-1], 
                                            all_soft_labels=all_soft_labels)
                elif 'quad' in exp_name:
                    representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                                            coarse_targets_map, idx, num_train_batches, 
                                            last_train_batch_size, num_classes[-1], 
                                            lambda_s=lambda_s, criterion_quad=criterion_quad)
                else:
                    representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                                            coarse_targets_map, idx, num_train_batches, 
                                            last_train_batch_size, num_classes[-1])
                
                if CPCC:
                    loss_cpcc = lamb * criterion_cpcc(representation, target_fine)
                    loss = loss_ce + loss_cpcc
                    train_losses_cpcc.append(loss_cpcc)
                elif group:
                    loss_group = criterion_group(model.module.fc.weight, model.module.fc.bias)
                    loss = loss_ce + loss_group
                    train_losses_group.append(loss_group)
                else:
                    loss = loss_ce
                train_losses_ce.append(loss_ce)
                
                loss.backward()
                optimizer.step()
            
                prob_fine = F.softmax(output_fine,dim=1)
                pred_fine = prob_fine.argmax(dim=1)
                acc_fine = pred_fine.eq(target_fine).flatten().tolist()
                train_fine_accs.extend(acc_fine)

                prob_coarse = get_layer_prob_from_fine(prob_fine, coarse_targets_map)
                pred_coarse = prob_coarse.argmax(dim=1)
                acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                train_coarse_accs.extend(acc_coarse)

                if idx % 100 == 1:
                    print(f"Train Loss: {loss}, Acc_fine: {sum(train_fine_accs)/len(train_fine_accs)}")
            
            scheduler.step()
            
            model.eval() 
            test_fine_accs = []
            test_coarse_accs = []
            test_losses_ce = []
            test_losses_cpcc = []
            test_losses_group = []
            
            with torch.no_grad():
                for idx, (data, _, target_coarse, _, target_fine) in enumerate(test_loader):
                    data = data.to(device)
                    target_coarse = target_coarse.to(device)
                    target_fine = target_fine.to(device)
                    
                    if 'soft' in exp_name:
                        representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                                                coarse_targets_map, idx, num_test_batches, 
                                                last_test_batch_size, num_classes[-1], 
                                                all_soft_labels=all_soft_labels)
                    elif 'quad' in exp_name:
                        representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                                                coarse_targets_map, idx, num_test_batches, 
                                                last_test_batch_size, num_classes[-1], 
                                                lambda_s=lambda_s, criterion_quad=criterion_quad)
                    else:
                        representation, output_fine, loss_ce = get_different_loss(exp_name, model, data, criterion_ce, target_fine, target_coarse, 
                                                coarse_targets_map, idx, num_test_batches, 
                                                last_test_batch_size, num_classes[-1])
                    
                    if CPCC:
                        loss_cpcc = lamb * criterion_cpcc(representation, target_fine)
                        loss = loss_ce + loss_cpcc
                        test_losses_cpcc.append(loss_cpcc)
                    elif group:
                        loss_group = criterion_group(model.module.fc.weight, model.module.fc.bias)
                        loss = loss_ce + loss_group
                        test_losses_group.append(loss_group)
                    else:
                        loss = loss_ce
                    test_losses_ce.append(loss_ce)
                    
                    prob_fine = F.softmax(output_fine,dim=1)
                    pred_fine = prob_fine.argmax(dim=1)
                    acc_fine = pred_fine.eq(target_fine).flatten().tolist()
                    test_fine_accs.extend(acc_fine)

                    prob_coarse = get_layer_prob_from_fine(prob_fine, coarse_targets_map)
                    pred_coarse = prob_coarse.argmax(dim=1)
                    acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                    test_coarse_accs.extend(acc_coarse)
            
            t_end = datetime.now()
            t_delta = (t_end-t_start).total_seconds()
            print(f"Val loss_ce: {sum(test_losses_ce)/len(test_losses_ce)}, Acc_fine: {sum(test_fine_accs)/len(test_fine_accs)}")
            print(f"Epoch {epoch} takes {t_delta} sec.")

            log_dict = {"train_fine_acc":sum(train_fine_accs)/len(train_fine_accs),
                        "train_losses_ce":sum(train_losses_ce)/len(train_losses_ce),
                        "val_fine_acc":sum(test_fine_accs)/len(test_fine_accs),
                        "val_losses_ce":sum(test_losses_ce)/len(test_losses_ce),
                    }
            
            if CPCC:
                log_dict["train_losses_cpcc"] = sum(train_losses_cpcc)/len(train_losses_cpcc)
                log_dict["val_losses_cpcc"] = sum(test_losses_cpcc)/len(test_losses_cpcc)
            if group:
                log_dict["train_losses_group"] = sum(train_losses_group)/len(train_losses_group)
                log_dict["val_losses_group"] = sum(test_losses_group)/len(test_losses_group)
            
            wandb.log(log_dict)
        
        torch.save(model.state_dict(), out_dir)
        wandb.finish()
        
        return model

    def curriculum_pretrain(train_loader : DataLoader, test_loader : DataLoader, 
                        device : torch.device, save_dir : str, seed : int, 
                        split : str, CPCC : bool, exp_name : str, epochs : int, 
                        task : str) -> nn.Module:
        '''
        Main train/test loop for curriculum learning.
        Args:
            train_loader : dataset train loader.
            test_loader : dataset test loader
            device : cpu/gpu
            save_dir : directory to save model checkpoint
            seed : random state
            split : split/full
            CPCC : use CPCC as a regularizer or not
            exp_name : curriculum or curriculumCPCC
            epochs : number of training iterations, note there are extra epochs for curriculum
            compared to other experiments
        '''
        assert 'curriculum' in exp_name, 'Invalid experiment'
        if task == 'in':
            dataset = train_loader.dataset.dataset
        else:
            dataset = train_loader.dataset
        num_fine_classes = [len(dataset.fine_names)]
        num_coarse_classes = [len(dataset.coarse_names)]
        coarse_targets_map = dataset.coarse_map

        split = split + task

        if CPCC:
            exp_name = exp_name + 'CPCC'


        init_config = {"dataset":dataset_name,
                    "exp_name":exp_name,
                    "_batch_size":train_loader.batch_size,
                    "epochs":epochs,
                    "_optimizer":'SGD',
                    "_scheduler":'StepLR',
                    "seed":seed,
                    "save_dir":save_dir,
                    "_num_workers":train_loader.num_workers,
                    "split":split,
                    "lamb":lamb,
                    "breeds_setting":breeds_setting,
                    }
        
        optim_config, scheduler_config = hyper['optimizer'], hyper['scheduler']
        config = {**init_config, **optim_config, **scheduler_config}  
        
        criterion_ce = nn.CrossEntropyLoss().to(device)
        criterion_cpcc = CPCCLoss(dataset, cpcc_layers, cpcc_metric).to(device)

        with open(save_dir+'/config.json', 'w') as fp:
            json.dump(config, fp, sort_keys=True, indent=4)
        
        out_dir = save_dir+f"/{split}_seed{seed}.pth" # fine model

        if os.path.exists(out_dir):
            print("Skipped.")
            return
            # msg = input("File Exists. Override the model? (y/n)")
            # if msg.lower() == 'n':
            #     print("Skipped.")
            #     return
            # elif msg.lower() == 'y':
            #     print("Retrain model.")
        
        # Step 1: train 20% epochs for coarse class only
        model = init_model(dataset_name, num_coarse_classes, device)
        optimizer, scheduler = init_optim_schedule(model, hyper)

        for epoch in range(int(epochs*0.2)):
            t_start = datetime.now()
            # ============== Stage 1 Train =================
            model.train()
            train_coarse_accs = []
            train_coarse_losses = []
            for idx, (data, _, target_coarse, _, _)  in enumerate(train_loader):
                data = data.to(device)
                target_coarse = target_coarse.to(device)
                optimizer.zero_grad()
                _, output_coarse = model(data) # we only add CPCC for the second stage training on fine level, no need for representation
                loss_ce = criterion_ce(output_coarse, target_coarse)
                loss_ce.backward()
                optimizer.step()
                prob_coarse = F.softmax(output_coarse,dim=1)
                pred_coarse = prob_coarse.argmax(dim=1, keepdim=False)
                acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                train_coarse_accs.extend(acc_coarse)
                train_coarse_losses.append(loss_ce)
                if idx % 100 == 1:
                    print(f"Train Loss: {loss_ce}, Acc_coarse: {sum(train_coarse_accs)/len(train_coarse_accs)}")
            scheduler.step()
            # ============== Stage 2 Test =================
            model.eval() 
            test_coarse_losses = []
            test_coarse_accs = []
            with torch.no_grad():
                for (data, _, target_coarse, _, _) in test_loader:
                    data = data.to(device)
                    target_coarse = target_coarse.to(device)
                    _, output_coarse = model(data)
                    loss_ce = criterion_ce(output_coarse, target_coarse)
                    test_coarse_losses.append(loss_ce)
                    prob_coarse = F.softmax(output_coarse,dim=1)
                    pred_coarse = prob_coarse.argmax(dim=1, keepdim=False)
                    acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                    test_coarse_accs.extend(acc_coarse)
            t_end = datetime.now()
            t_delta = (t_end-t_start).total_seconds()
            print(f"Val loss: {sum(test_coarse_losses)/len(test_coarse_losses)}, Acc_coarse: {sum(test_coarse_accs)/len(test_coarse_accs)}")
            print(f"Epoch {epoch} takes {t_delta} sec.")
        torch.save(model.state_dict(), save_dir+f"/{split}_coarse_seed{seed}.pth")


        wandb.init(project=f"{dataset_name}_onestage_pretrain", # reset, log to one stage
                entity="structured_task",
                name=datetime.now().strftime("%m%d%Y%H%M%S"),
                config=config)
        
        # Step 2: train 80% epochs for fine class only
        model = init_model(dataset_name, num_fine_classes, device)
        model_dict = model.state_dict()
        coarse_dict = {k: v for k, v in torch.load(save_dir+f"/{split}_coarse_seed{seed}.pth").items() if (k in model_dict) and ('fc' not in k)}
        model_dict.update(coarse_dict) 
        model.load_state_dict(model_dict)
        # reset optimizer and scheduler
        optimizer, scheduler = init_optim_schedule(model, hyper)
        for epoch in range(int(epochs*0.8)):
            t_start = datetime.now()
            # ============== Stage 2 Train =================
            model.train()
            train_fine_accs = []
            train_coarse_accs = []
            train_losses_ce = []
            train_losses_cpcc = []

            for idx, (data, _, target_coarse, _, target_fine) in enumerate(train_loader):
                data = data.to(device)
                target_fine = target_fine.to(device)
                target_coarse = target_coarse.to(device)

                optimizer.zero_grad()
                representation, output_fine = model(data)
                loss_ce = criterion_ce(output_fine, target_fine)

                if CPCC:
                    loss_cpcc = lamb * criterion_cpcc(representation, target_fine)
                    loss = loss_ce + loss_cpcc
                    train_losses_cpcc.append(loss_cpcc)
                else:
                    loss = loss_ce
                train_losses_ce.append(loss_ce)

                loss.backward()
                optimizer.step()

                prob_fine = F.softmax(output_fine,dim=1)
                pred_fine = prob_fine.argmax(dim=1)
                acc_fine = pred_fine.eq(target_fine).flatten().tolist()
                train_fine_accs.extend(acc_fine)

                prob_coarse = get_layer_prob_from_fine(prob_fine, coarse_targets_map)
                pred_coarse = prob_coarse.argmax(dim=1)
                acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                train_coarse_accs.extend(acc_coarse)

                if idx % 100 == 1:
                    print(f"Train Loss: {loss}, Acc_fine: {sum(train_fine_accs)/len(train_fine_accs)}")

            scheduler.step()

            # ============== Stage 2 Test =================
            model.eval() 
            test_fine_accs = []
            test_coarse_accs = []
            test_losses_ce = []
            test_losses_cpcc = []

            with torch.no_grad():
                for idx, (data, _, target_coarse, _, target_fine) in enumerate(test_loader):
                    data = data.to(device)
                    target_coarse = target_coarse.to(device)
                    target_fine = target_fine.to(device)

                    representation, output_fine = model(data)
                    loss_ce = criterion_ce(output_fine, target_fine)

                    if CPCC:
                        loss_cpcc = lamb * criterion_cpcc(representation, target_fine)
                        loss = loss_ce + loss_cpcc
                        test_losses_cpcc.append(loss_cpcc)
                    else:
                        loss = loss_ce
                    test_losses_ce.append(loss_ce)

                    prob_fine = F.softmax(output_fine,dim=1)
                    pred_fine = prob_fine.argmax(dim=1)
                    acc_fine = pred_fine.eq(target_fine).flatten().tolist()
                    test_fine_accs.extend(acc_fine)

                    prob_coarse = get_layer_prob_from_fine(prob_fine, coarse_targets_map)
                    pred_coarse = prob_coarse.argmax(dim=1)
                    acc_coarse = pred_coarse.eq(target_coarse).flatten().tolist()
                    test_coarse_accs.extend(acc_coarse)

            t_end = datetime.now()
            t_delta = (t_end-t_start).total_seconds()
            print(f"Val loss_ce: {sum(test_losses_ce)/len(test_losses_ce)}, Acc_fine: {sum(test_fine_accs)/len(test_fine_accs)}")
            print(f"Epoch {epoch} takes {t_delta} sec.")

            log_dict = {"train_fine_acc":sum(train_fine_accs)/len(train_fine_accs),
                        "train_losses_ce":sum(train_losses_ce)/len(train_losses_ce),
                        "val_fine_acc":sum(test_fine_accs)/len(test_fine_accs),
                        "val_losses_ce":sum(test_losses_ce)/len(test_losses_ce),
                    }
            
            if CPCC:
                log_dict["train_losses_cpcc"] = sum(train_losses_cpcc)/len(train_losses_cpcc)
                log_dict["val_losses_cpcc"] = sum(test_losses_cpcc)/len(test_losses_cpcc)
            
            wandb.log(log_dict)

        torch.save(model.state_dict(), out_dir)
        wandb.finish()
        return model

    if 'curriculum' in exp_name:
        if dataset_name == 'CIFAR':
            extra_epochs = int(epochs * 1.25) # default : 250
        elif dataset_name == 'MNIST':
            extra_epochs = int(epochs * 1.2)
        elif dataset_name == 'BREEDS':
            if breeds_setting == 'living17' or breeds_setting == 'nonliving26':
                extra_epochs = int(epochs * 1.2)
            elif breeds_setting == 'entity13' or breeds_setting == 'entity30':
                extra_epochs = int(epochs * 1.25)
        curriculum_pretrain(train_loader, test_loader, device, save_dir, seed, split, CPCC, exp_name, extra_epochs, task)
    else:
        one_stage_pretrain(train_loader, test_loader, device, save_dir, seed, split, CPCC, exp_name, epochs, task)
    return

def downstream_transfer(save_dir : str, seed : int, device : torch.device, 
                        batch_size : int, level : str, CPCC : bool,
                        exp_name : str, num_workers : int, task : str, 
                        dataset_name : str, case : int, breeds_setting : str,
                        hyper, epochs) -> nn.Module:
    '''
        Transfer to target sets on new level.
    '''
    train_loader, test_loader = make_kshot_loader(num_workers, batch_size, 1, level, 
                                                task, seed, dataset_name, case, breeds_setting) # we use one shot on train set
    dataset = train_loader.dataset.dataset # loader contains Subset
    if level == 'fine':
        num_classes = len(dataset.fine_names)
    elif level == 'mid':
        num_classes = len(dataset.mid_names)
    elif level == 'coarse':
        num_classes = len(dataset.coarse_names)
    elif level == 'coarsest':
        num_classes = len(dataset.coarsest_names)
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    criterion = nn.CrossEntropyLoss().to(device) # no CPCC in downstream task
    
    model = init_model(dataset_name, [num_classes], device)
    
    model_dict = model.state_dict()
    # load pretrained seed 0, call this function #seeds times
    trained_dict = {k: v for k, v in torch.load(save_dir+f"/split{task}_seed0.pth").items() if (k in model_dict) and ("fc" not in k)}
    model_dict.update(trained_dict) 
    model.load_state_dict(model_dict)

    for param in model.parameters(): # Freeze Encoder, fit last linear layer
        param.requires_grad = False
    model.module.fc = nn.Linear(model.module.out_features, num_classes).to(device)

    if CPCC:
        exp_name = exp_name + 'CPCC'
    
    init_config = {"_batch_size":train_loader.batch_size,
                    "epochs":epochs,
                    "_optimizer":'SGD',
                    "_scheduler":'StepLR',
                    "seed":seed,
                    "save_dir":save_dir,
                    "exp_name":exp_name,
                    "_num_workers":train_loader.num_workers,
                    "new_level":level,
                    "task":task,
                    "breeds_setting":breeds_setting,
                    }
    
    optim_config, scheduler_config = hyper['optimizer'], hyper['scheduler']
    config = {**init_config, **optim_config, **scheduler_config}  

    wandb.init(project=f"{dataset_name}-subpop", # {dataset_name}-subpop/new-level-tuning
               entity="structured_task",
               name=datetime.now().strftime("%m%d%Y%H%M%S"),
               config=config
    )
    
    optimizer, scheduler = init_optim_schedule(model, hyper)

    for epoch in range(epochs):
        t_start = datetime.now()
        model.eval() 
        test_top1 = 0
        test_top2 = 0
        test_losses = []
        
        with torch.no_grad():
            for (data, target_coarsest, target_coarse, target_mid, target_fine) in test_loader:
                data = data.to(device)
                target_coarse = target_coarse.to(device)
                target_fine = target_fine.to(device)
                target_mid = target_mid.to(device)
                target_coarsest = target_coarsest.to(device)
                
                if level == 'coarsest':
                    target = target_coarsest
                elif level == 'mid':
                    target = target_mid
                elif level == 'coarse':
                    target = target_coarse
                elif level == 'fine':
                    target = target_fine

                _, output = model(data)
                loss = criterion(output, target)
                test_losses.append(loss)

                prob = F.softmax(output,dim=1)

                # top 1
                pred1 = prob.argmax(dim=1, keepdim=False) 
                top1_correct = pred1.eq(target).sum()
                test_top1 += top1_correct

                # top 2
                pred2 = (prob.topk(k=2, dim=1)[1]).T # 5 * batch_size
                target_reshaped = target.unsqueeze(0).expand_as(pred2)
                top2_correct = pred2.eq(target_reshaped).sum() 
                test_top2 += top2_correct
        
        print(f"Val loss: {sum(test_losses)/len(test_losses)}, Top1_{level}: {test_top1/test_size} "
              f"Top2_{level} : {test_top2/test_size}")

        model.train()
        train_top1 = 0
        train_top2 = 0
        train_losses = []
        for idx, (data, target_coarsest, target_coarse, target_mid, target_fine) in enumerate(train_loader):
            data = data.to(device)
            target_coarse = target_coarse.to(device)
            target_fine = target_fine.to(device)
            target_mid = target_mid.to(device)
            target_coarsest = target_coarsest.to(device)
            
            if level == 'coarsest':
                target = target_coarsest
            elif level == 'mid':
                target = target_mid
            elif level == 'coarse':
                target = target_coarse
            elif level == 'fine':
                target = target_fine

            optimizer.zero_grad()
            _, output = model(data)
            loss = criterion(output, target)
            train_losses.append(loss)
    
            loss.backward()
            optimizer.step()
            
            prob = F.softmax(output,dim=1)

            pred1 = prob.argmax(dim=1, keepdim=False) 
            top1_correct = pred1.eq(target).sum()
            train_top1 += top1_correct

            pred2 = (prob.topk(k=2, dim=1)[1]).T 
            target_reshaped = target.unsqueeze(0).expand_as(pred2)
            top2_correct = pred2.eq(target_reshaped).sum() 
            train_top2 += top2_correct
            
            if idx % 100 == 0:
                print(f"Train loss: {sum(train_losses)/len(train_losses)}, Top1_{level}: {train_top1/train_size} "
                      f"Top2_{level} : {train_top2/train_size}")

        scheduler.step()
        
        t_end = datetime.now()
        t_delta = (t_end-t_start).total_seconds()
        print(f"Epoch {epoch} takes {t_delta} sec.")
        
        wandb.log({
            "train_top1":train_top1/train_size,
            "train_top2":train_top2/train_size,
            "train_losses":sum(train_losses)/len(train_losses),
            "val_top1":test_top1/test_size,
            "val_top2":test_top2/test_size,
            "val_losses":sum(test_losses)/len(test_losses),
        })
    
    torch.save(model.state_dict(), save_dir+f"/down{task}_{level}_seed{seed}.pth")
    wandb.finish()
    
    return model

def downstream_zeroshot(seeds : int , save_dir, split, task, source_train_loader, 
                        target_test_loader, levels : List[str], exp_name, device, 
                        dataset_name):
    
    # If all classes in test loader at level are already seen in train loader
    # try to use train set's label hierarchy for zero shot classification
    
    train_dataset = source_train_loader.dataset 
    test_dataset = target_test_loader.dataset
    
    if cpcc:
        exp_name = exp_name + 'CPCC'
    zero_shot = {'exp_name' : exp_name}
    
    for level in levels:
        
        level_res = []
        
        if level == 'fine':
            train_classes, test_classes = train_dataset.fine_names, test_dataset.fine_names
        elif level == 'mid':
            train_classes, test_classes = train_dataset.mid_names, test_dataset.mid_names
            layer_fine_map = train_dataset.mid_map
        elif level == 'coarse':
            train_classes, test_classes = train_dataset.coarse_names, test_dataset.coarse_names
            layer_fine_map = train_dataset.coarse_map
        elif level == 'coarsest':
            train_classes, test_classes = train_dataset.coarsest_names, test_dataset.coarsest_names
            layer_fine_map = train_dataset.coarsest_map
        
        assert (train_classes == test_classes), f'Zero shot invalid for {level}.'
        
        for seed in range(seeds):
            if 'MTL' in exp_name:
                model = init_model(dataset_name, [len(train_dataset.coarse_names),len(train_dataset.fine_names)], device)
            else:
                model = init_model(dataset_name, [len(train_dataset.fine_names)], device)
            model.load_state_dict(torch.load(save_dir + f'/{split}{task}_seed{seed}.pth'))
            model.eval()
            
            layer_accs = []
            with torch.no_grad():
                for (data, target_coarsest, target_coarse, target_mid, target_fine) in target_test_loader:
                    data = data.to(device)
                    target_coarsest = target_coarsest.to(device)
                    target_coarse = target_coarse.to(device)
                    target_mid = target_mid.to(device)
                    target_fine = target_fine.to(device)

                    if level == 'coarsest':
                        target_layer = target_coarsest
                    elif level == 'coarse':
                        target_layer = target_coarse
                    elif level == 'mid':
                        target_layer = target_mid
                    elif level == 'fine':
                        target_layer = target_fine
                    
                    if 'MTL' in exp_name:
                        _, _, output_fine = model(data)
                    else:
                        _, output_fine = model(data)
                    prob_fine = F.softmax(output_fine,dim=1)
                    pred_fine = prob_fine.argmax(dim=1, keepdim=False)
                    if level == 'fine':
                        pred_layer = pred_fine
                    else:
                        prob_layer = get_layer_prob_from_fine(prob_fine, layer_fine_map)
                        pred_layer = prob_layer.argmax(dim=1, keepdim=False)
                    acc_layer = list(pred_layer.eq(target_layer).flatten().cpu().numpy())
                    layer_accs.extend(acc_layer)
            
            level_res.append(sum(layer_accs)/len(layer_accs))
        
        zero_shot[level] = {'value' : level_res, 'mean' : np.average(level_res), 'std' : np.std(level_res)}
    
    with open(save_dir+f'/{split}{task}_zero_shot.json', 'w') as fp:
        json.dump(zero_shot, fp, indent=4)
    print(zero_shot)
    
    return layer_accs

def feature_extractor(dataloader : DataLoader, split : str, task : str, dataset_name : str, seed : int):
    dataset = dataloader.dataset
    model = init_model(dataset_name, [len(dataset.fine_names)], device)

    model_dict = model.state_dict()
    ckpt_dict = {k: v for k, v in torch.load(save_dir+f"/{split}{task}_seed{seed}.pth").items() if (k in model_dict) and ('fc' not in k)}
    model_dict.update(ckpt_dict) 
    model.load_state_dict(model_dict)
    model.module.fc = nn.Identity()

    features = []
    targets_fine = []
    with torch.no_grad():
        for item in dataloader:
            data = item[0]
            target_fine = item[-1]
            data = data.to(device)
            target_fine = target_fine.to(device)
            feature = model(data)[0]
            features.append(feature.cpu().detach().numpy())
            targets_fine.append(target_fine.cpu().detach().numpy())
    features = np.concatenate(features,axis=0)
    labels = np.concatenate(targets_fine,axis=0)
    return (features, labels)

def ood_detection(seeds : int, dataset_name : str, exp_name : str):
    '''
        Set CIFAR10 as the outlier of CIFAR100.
        Credit: https://github.com/boschresearch/rince/blob/cifar/out_of_dist_detection.py
    '''
    import cifar.data
    assert dataset_name == 'CIFAR', 'Invalid dataset for OOD detection'

    in_train_loader, in_test_loader = cifar.data.make_dataloader(num_workers, batch_size, 'full')
    _, out_test_loader = cifar.data.make_dataloader(num_workers, batch_size, 'outlier')
    oods = []
    out = {}
    for seed in range(seeds):
        # compute features
        in_train_features, in_train_labels = feature_extractor(in_train_loader, 'full', '', dataset_name, seed)
        in_test_features, _ = feature_extractor(in_test_loader, 'full', '', dataset_name, seed)
        out_test_features, _ = feature_extractor(out_test_loader, 'full', '', dataset_name, seed)
        print("Features successfully loaded.")

        features_outlier = np.concatenate([out_test_features, in_test_features], axis=0)
        labels = np.concatenate([np.zeros(out_test_features.shape[0], ),
                                np.ones(in_test_features.shape[0], )], axis=0)

        gms = {}
        posteriors = np.zeros((features_outlier.shape[0], len(np.unique(in_train_labels))))
        for i, label in enumerate(np.unique(in_train_labels)):
            means = np.mean(in_train_features[in_train_labels == label, :], axis=0).reshape((1, -1))
            gms[str(label)] = GaussianMixture(1, random_state=seed, means_init=means).fit(
                in_train_features[in_train_labels == label, :]) 
            posteriors[:, i] = gms[str(label)].score_samples(features_outlier)

        max_score = np.max(posteriors, axis=1)
        auroc = roc_auc_score(labels, max_score) # try different thresholds for max score
        oods.append(auroc)
    if cpcc:
        exp_name = exp_name + 'CPCC'
    out['exp_name'] = exp_name
    out['OOD'] = oods
    out['mean'] = np.average(oods)
    out['std'] = np.std(oods)
    with open(save_dir+'/OOD.json', 'w') as fp:
        json.dump(out, fp, indent=4)
    print(out)
    return oods

def main():
    
    # Train
    
    for seed in range(seeds):
        seed_everything(seed)
        if split == 'split':
            # pretrain
            hyper = load_params(dataset_name, 'pre', breeds_setting=breeds_setting)
            epochs = hyper['epochs']
            
            if task == 'sub':
                train_loader, test_loader = make_dataloader(num_workers, batch_size, 'sub_split_pretrain', dataset_name, case, breeds_setting)
            elif task == 'in':
                train_loader, test_loader = make_dataloader(num_workers, batch_size, 'in_split_pretrain', dataset_name, case, breeds_setting)
            pretrain_objective(train_loader, test_loader, device, save_dir, seed, split, cpcc, exp_name, epochs, task, dataset_name, breeds_setting, hyper)
            
            # down
            for level in ['mid','fine']: 
                hyper = load_params(dataset_name, 'down', level, breeds_setting)
                epochs = hyper['epochs']
                downstream_transfer(save_dir, seed, device, batch_size, level, cpcc, exp_name, num_workers, task, dataset_name, case, breeds_setting, hyper, epochs)
        
        elif split == 'full': 
            hyper = load_params(dataset_name, 'pre', breeds_setting=breeds_setting)
            epochs = hyper['epochs']
            train_loader, test_loader = make_dataloader(num_workers, batch_size, 'full', dataset_name, case, breeds_setting) # full
            pretrain_objective(train_loader, test_loader, device, save_dir, seed, split, cpcc, exp_name, epochs, task, dataset_name, breeds_setting, hyper)

    # Eval: zero-shot/ood

    if task == 'sub':
        if dataset_name == 'MNIST':
            levels = ['coarse']
        else:
            levels = ['coarsest','coarse'] 
        train_loader, test_loader = make_dataloader(num_workers, batch_size, f'{task}_split_zero_shot', dataset_name, case, breeds_setting)
    elif task == '': # full
        if dataset_name == 'MNIST':
            levels = ['coarse','mid','fine'] 
        else:
            levels = ['coarsest','coarse','mid','fine']
        train_loader, test_loader = make_dataloader(num_workers, batch_size, 'full', dataset_name, case, breeds_setting)
    
    downstream_zeroshot(seeds, save_dir, split, task, train_loader, test_loader, levels, exp_name, device, dataset_name)

    if dataset_name == 'CIFAR':
        ood_detection(seeds, dataset_name, exp_name)
    return

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", required=True, help=r'datetime.now().strftime("%m%d%Y%H%M%S")') # TODO: better initialization to avoid exp_name mistmatch mistakes
    parser.add_argument("--dataset", required=True, help='MNIST/CIFAR/BREEDS')
    parser.add_argument("--exp_name", required=True, help='ERM/MTL/Curriculum/sumloss/HXE/soft/quad')
    parser.add_argument("--split", required=True, help='split/full')
    parser.add_argument("--task", default='', help='in/sub')
    parser.add_argument("--cpcc", required=True, type=int, help='0/1')
    parser.add_argument("--cpcc_metric", default='l2', type=str, help='distance metric in CPCC, can be l2/l1/poincare')
    parser.add_argument("--cpcc_list", nargs='+', default=['coarse'], help='ex: --cpcc-list mid coarse, for 3 layer cpcc')
    parser.add_argument("--group", default=0, type=int, help='0/1, grouplasso')
    parser.add_argument("--case", type=int, help='Type of MNIST, 0/1')

    parser.add_argument("--lamb",type=float,default=1,help='strength of CPCC regularization')
    
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seeds", type=int,default=5)    
    
    args = parser.parse_args()
    timestamp = args.timestamp
    exp_name = args.exp_name
    dataset_name = args.dataset
    split = args.split
    task = args.task
    cpcc = args.cpcc
    cpcc_metric = args.cpcc_metric
    cpcc_layers = args.cpcc_list
    case = args.case
    group = args.group

    num_workers = args.num_workers
    batch_size = args.batch_size
    seeds = args.seeds
    lamb = args.lamb

    root = f'/data/common/cindy2000_sh/hierarchy_results/{dataset_name}'
    save_dir = root + '/' + timestamp 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(root + f'/{timestamp}'):
        os.makedirs(root + f'/{timestamp}')

    if dataset_name == 'BREEDS':
        for breeds_setting in ['living17','entity13','entity30','nonliving26']:
            save_dir = root + '/' + timestamp + '/' + breeds_setting
            main()
    
    else:
        breeds_setting = None
        main()
    