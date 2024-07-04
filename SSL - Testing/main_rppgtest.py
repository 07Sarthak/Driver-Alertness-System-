import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from data_rppg import get_dataset_test  # Assuming you have a function to load your test dataset
from net_full import Mynet  # Assuming you have defined your model architecture
from loss import FRL, FAL, FCL  # Assuming you have defined your loss functions
import numpy as np
import matplotlib.pyplot as plt
from testread import read_groundtruth
import os
import math
from scipy.signal import find_peaks

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test the trained model')
parser.add_argument('--model_path', type=str, required=True, default='./weights/rppg_model_epoch_29.pth', help='Path to the trained model weights')
parser.add_argument('--batchSize', type=int, default=1, help='Batch size for testing')
parser.add_argument('--threads', type=int, default=0, help='Number of threads for data loader')
parser.add_argument('--num_negative', type=int, default=4, help='Number of negative samples')
parser.add_argument('--video_length', type=int, default=1801, help='Video length')
parser.add_argument('--num_expert', type=int, default=9, help='Number of experts')
parser.add_argument('--file_list', type=str, default='testlist.txt', help='Path to the test file list')
args = parser.parse_args()

# Check if GPU is available
cuda = torch.cuda.is_available()
print('helo')
# Load the trained model
model = Mynet(base_filter=64, video_length=args.video_length, num_expert=args.num_expert)
model.load_state_dict(torch.load(args.model_path))
if cuda:
    model = model.cuda()

# Set model to evaluation mode
model.eval()
print('hi')

# Define loss functions
# criterion = nn.MSELoss()
# criterion2 = FRL(Fs=30, min_hr=40, max_hr=180)
# criterion3 = FAL(Fs=30, high_pass=2.5, low_pass=0.4)
# criterion4 = FCL(Fs=30, high_pass=2.5, low_pass=0.4)
# if cuda:
#     criterion = criterion.cuda()
#     criterion2 = criterion2.cuda()
#     criterion3 = criterion3.cuda()
#     criterion4 = criterion4.cuda()

# Prepare the test dataset
test_set = get_dataset_test(args.file_list, args.num_negative)
test_data_loader = DataLoader(dataset=test_set, num_workers=args.threads, batch_size=args.batchSize, shuffle=False)

# Initialize variables to store evaluation metrics
# total_loss_rec = 0.0
# total_loss_frl = 0.0
# total_loss_fal = 0.0
# total_loss_fcl = 0.0
# total_rmse_loss = 0.0
# total_samples = 0

# def mse_loss(input, negative_arr):
#     l_mse = 0
#     for i in range(len(negative_arr)):
#         l_mse_negative = criterion(input, negative_arr[i])
#         l_mse += l_mse_negative
#     return l_mse/len(negative_arr)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

def min_max_normalize(data):
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

# Example usage:




# def checkpoint(epoch):
#     model_out_path = opt.save_folder + "rppg_model_epoch_{}.pth".format(epoch)
#     torch.save(model.state_dict(), model_out_path)
#     print("Checkpoint saved to {}".format(model_out_path))
count_file = 0
# Iterate over test data and perform inference
for batch, file_path in test_data_loader:
    # Extract input and target from the batch
    input = batch
    print('input size', input.shape)

#     # Move data to GPU if available
    if cuda:
        input = input.cuda()
        # positive1 = positive1.cuda()
        # positive2 = positive2.cuda()
        # neighbor1 = neighbor1.cuda()
        # neighbor2 = neighbor2.cuda()
        # neighbor3 = neighbor3.cuda()
        # ratio_array = ratio_array.cuda()

    # Perform forward pass
    print(file_path)
    frame_list = os.path.join(file_path[0],'frames')
    list = os.listdir(frame_list)
    video_length=len(list)
    with torch.no_grad():
        input_rppg  = model(input, video_length)
    
    time, truth = read_groundtruth(file_path)
    input_rppg = input_rppg.cpu().detach().numpy()
    # print(input_rppg, '\n', truth)
    input_rppg_normalized = min_max_normalize(input_rppg[0])
    truth_normalized = min_max_normalize(truth)
    time_x1 = range(len(input_rppg[0]))
    time_x2 = range(len(truth))
    # print(time_x1)
    # print(time_x2)
    print(input_rppg_normalized[1000:1300])
    plt.plot(input_rppg_normalized[1000:1300], label = 'Predicted')
    plt.plot(truth_normalized[1000:1300], label = 'truth')
    plt.title(f'rPPG vs Time graph - {count_file}')
    plt.xlabel('Time')
    plt.ylabel('rPPG signals')
    plt.legend()
    plt.grid()
    print('helllo')
    plt.savefig(f'plot_{count_file}.png')
    count_file += 1
    plt.clf()
    input_sample1 = input_rppg_normalized[:150]
    truth_sample1 = truth_normalized[:150]
    truth_sample1 = np.array(truth_sample1)
    peak_pred, _ = find_peaks(input_sample1,height=0)
    peak_real,_ = find_peaks(truth_sample1,height=0)
    ibis_pred = np.diff(peak_pred)
    ibis_real = np.diff(peak_real)
    BPM_pred = 60 / (ibis_pred/30)
    BPM_real = 60 / (ibis_real/30)
    print(len(BPM_pred),' - Predicted')
    print(len(BPM_real),' - Real')
    # mae = truth_sample1 - input_sample1
    # rmse = []
    # for i in mae:
    #     i = abs(i)
    #     rmse.append(i*i)
    # rmse = np.array(rmse)
    # mae = np.mean(mae)
    # rmse = np.mean(rmse)
    # rmse = math.sqrt(rmse)
    # print('RMSE Loss:',rmse)
    # print('MAE Loss:',mae)
    break
# we know that UBFC datasets record at 30 fps so we can find the exact location of the
# ground truth given in the predicted rPPG.
#         # Compute losses
#         loss_rec = mse_loss(input, negative_arr)
#         loss_frl = criterion2(neg_rppgarr, pos_rppg1, pos_rppg2, ratio_array)
#         loss_fal = criterion3(pos_rppg1, pos_rppg2, neighbor_rppg1, neighbor_rppg2, neighbor_rppg3)
#         loss_fcl = criterion4(neg_rppgarr, pos_rppg1, pos_rppg2)

#         # Compute RMSE loss
#         #rmse_loss = torch.sqrt(criterion(input, negative_arr))

#         # Update total losses and sample count
#         total_loss_rec += loss_rec.item() * input.size(0)
#         total_loss_frl += loss_frl.item() * input.size(0)
#         total_loss_fal += loss_fal.item() * input.size(0)
#         total_loss_fcl += loss_fcl.item() * input.size(0)
#         #total_rmse_loss += rmse_loss.item() * input.size(0)
#         total_samples += input.size(0)

# # Calculate average losses
# avg_loss_rec = total_loss_rec / total_samples
# avg_loss_frl = total_loss_frl / total_samples
# avg_loss_fal = total_loss_fal / total_samples
# avg_loss_fcl = total_loss_fcl / total_samples
# #avg_rmse_loss = total_rmse_loss / total_samples

# # Print or log the evaluation results
# print(f'Average Loss (Reconstruction): {avg_loss_rec:.4f}')
# print(f'Average Loss (FRL): {avg_loss_frl:.4f}')
# print(f'Average Loss (FAL): {avg_loss_fal:.4f}')
# print(f'Average Loss (FCL): {avg_loss_fcl:.4f}')
# #print(f'Average RMSE Loss: {avg_rmse_loss:.4f}')

# # Optionally, save the evaluation results to a log file
# with open('evaluation_results.txt', 'w') as f:
#     f.write(f'Average Loss (Reconstruction): {avg_loss_rec:.4f}\n')
#     f.write(f'Average Loss (FRL): {avg_loss_frl:.4f}\n')
#     f.write(f'Average Loss (FAL): {avg_loss_fal:.4f}\n')
#     f.write(f'Average Loss (FCL): {avg_loss_fcl:.4f}\n')
#     #f.write(f'Average RMSE Loss: {avg_rmse_loss:.4f}\n')
