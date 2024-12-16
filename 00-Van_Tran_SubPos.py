'''
The vanilla Transforemr is the original VAN_Transformer model without any modification.

The difference between the vanilla VAN_Transformer and the vanilla TransBrainer is the data embedding method.

The vanilla VAN_Transformer uses the original data embedding method, which uses the token embedding and positional embedding.
However, the abalation study of masking method is also conducted in the vanilla VAN_Transformer too.

!! This version is the real original VAN_Transformer model.

'''

from Van_Transformer_Model import *
import pdb
import os
import argparse
import random
import pandas as pd
import numpy as np
from torch.nn.functional import mse_loss
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import savemat

def save_to_mat(input_data, output_data, save_path):
    """
    Save input and output data to a .mat file.

    Args:
        input_data (numpy.ndarray): Input data array.
        output_data (numpy.ndarray): Output data array.
        save_path (str): Path to save the .mat file.
    """
    mat_data = {
        'SC': input_data,
        'FC': output_data
    }
    savemat(save_path, mat_data)


seed = 42 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--dim_ffn', type=int, default=2048)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--out', type=str)
parser.add_argument('-gpu_id', '--gpu_id', type=int, default=1)

args = parser.parse_args()

# Hyperparameters
device = "cuda:" + str(args.gpu_id)
batch_size = args.batch_size
num_epochs = args.num_epochs
dim_feedforward = args.dim_ffn
learning_rate = args.lr
weight_decay = args.weight_decay
dropout = args.dropout

out = args.out

# Parameters
num_workers = 0
input_dim = 68

output_path_base = f"/outputpath"

output_path = os.path.join(output_path_base, out)
os.makedirs(output_path, exist_ok=True)

fold_results = {}

for fold in range(5):
    fold_output_path = os.path.join(output_path, f"Fold_{fold+1}")
    os.makedirs(fold_output_path, exist_ok=True)
    
    all_data = load_data(fold+1)
    data_mean = normlize_data(all_data)

    train_data_loader = get_loader(all_data, data_mean, True, False, batch_size, num_workers=num_workers)
    test_data_loader = get_loader(all_data, data_mean, False, True, batch_size, num_workers=num_workers)
    
    model = VAN_Transformer(input_dim=input_dim, dim_feedforward=dim_feedforward, dropout=dropout).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, 0.5, last_epoch=-1)
        
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        rscore_total = 0.0
        for i, (src_batch, tgt_batch) in enumerate(train_data_loader):
            src_batch = src_batch.to(device).float()
            tgt_batch = tgt_batch.to(device).float()

            dec_mask = get_attn_vanilla_decoder_mask(tgt_batch).to(device)
            
            outputs = model(src_batch, src_batch, dec_mask)
            mse_loss = criterion(outputs, tgt_batch)
            corr = pearson_correlation_coefficient(outputs, tgt_batch)

            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()

            running_loss += mse_loss.item()
            rscore_total += corr

        scheduler.step()

        avg_train_loss = running_loss / len(train_data_loader)
        avg_train_corr = rscore_total / len(train_data_loader)
        
        print(f"[Fold {fold+1} Train] Epoch {epoch+1}/{num_epochs} | MSE Loss: {avg_train_loss:.4f}, R score: {avg_train_corr:.4f}")
    
    model_path = os.path.join(fold_output_path, f"fold_{fold+1}_model.pth")
    torch.save(model.state_dict(), model_path)
    
    model.eval()
    test_loss = 0.0
    test_rscore = 0.0
    
    sample_index = 0
    
    for i, (src_batch, tgt_batch) in enumerate(test_data_loader):
        src_batch = src_batch.to(device).float()
        tgt_batch = tgt_batch.to(device).float()
        
        dec_mask = get_attn_vanilla_decoder_mask(tgt_batch).to(device)
        
        outputs = model(src_batch, src_batch, dec_mask)
        
        test_mse_loss = criterion(outputs, tgt_batch)
        test_corr = pearson_correlation_coefficient(outputs, tgt_batch)

        optimizer.zero_grad()
        test_mse_loss.backward()
        optimizer.step()

        test_loss += test_mse_loss.item()
        test_rscore += test_corr
        
        src_batch_np = src_batch.cpu().detach().numpy()
        outputs_np = outputs.cpu().detach().numpy()

        for j in range(len(src_batch_np)):
            sample_input = src_batch_np[j]
            sample_output = outputs_np[j]
            
            sample_save_path = os.path.join(fold_output_path, f"sample_{sample_index + 1}.mat")
            save_to_mat(sample_input, sample_output, sample_save_path)
            
            sample_index += 1
        
    avg_test_loss = test_loss / len(test_data_loader)
    avg_test_corr = test_rscore / len(test_data_loader)
    
    fold_results[f"Fold {fold+1}"] = {
        "test_mse": avg_test_loss,
        "test_rscore": avg_test_corr
    }
    
    print("***************************************************")
    print(f"* [Fold {fold+1} Test] MSE Loss: {avg_test_loss:.4f}, R score: {avg_test_corr:.4f} *")
    print("***************************************************")

print("-------------------------Fold Results----------------------")
for fold, results in fold_results.items():
    fold_results_file = os.path.join(output_path, "fold_results.txt")
    if not os.path.isfile(fold_results_file):
        with open(fold_results_file, "w") as file:
            file.write(f"Combination: num_epochs={num_epochs}, batch_size={batch_size}, dim_feedforward={dim_feedforward}, lr={learning_rate}, weight_decay={weight_decay}, dropout={dropout} | out={out}\n")
    print(f"{fold} | Test MSE Loss: {results['test_mse']:.4f}, R score: {results['test_rscore']:.4f}")
    with open(fold_results_file, "a") as file:
        file.write(f"{fold} | Test MSE Loss: {results['test_mse']:.4f}, R score: {results['test_rscore']:.4f}\n")
