import os
import numpy as np
import torch
import sys

# Add path to backend_driver_ass folder
backend_driver_ass_path = r'C:\Users\sarth\Desktop\Driver_allertness_P41A\Dual_GAN\backend_driver_ass'
sys.path.append(backend_driver_ass_path)
from model_folder import model
from torch.autograd import Variable
import torchvision.transforms.functional as transF
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from directory import sample_location
feature_map_path=os.path.join(sample_location,'video_5x5_ori.png')
ground_truth_path=os.path.join(sample_location,'ground_truth.txt')
test_wave=[]
test_gt=[]
def test(feature_map_path,ground_truth_path):
    GPU = '0'
    if torch.cuda.is_available():
        device = torch.device('cuda:' + GPU if torch.cuda.is_available() else 'cpu')  #
        print('on GPU')
    else:
        print('on CPU')


    def transform(image):
        image = transF.resize(image, size=(64, 256))
        image = transF.to_tensor(image)
        image=image/255
        image = transF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return image


    final_wave_pr=np.array([0])
    final_bvp_gt=np.array([0])
    feature_map_path = feature_map_path
    feature_map = cv2.imread(feature_map_path)
    for i in range(int(feature_map.shape[0]/256)):
        feature_map = cv2.imread(feature_map_path)
        feature_map = torch.tensor(feature_map)
        feature_map = feature_map.permute(1, 0, 2)
        feature_map = feature_map[:, 256*i:256*i+256, :]
        for c in range(feature_map.shape[2]):
            for r in range(feature_map.shape[0]):
                feature_map[r, :, c] = 225*((feature_map[r, :, c] - torch.min(feature_map[r, :, c])) / (
                                                                                                          torch.max(
                                                                                                              feature_map[r, :,
                                                                                                              c]) - torch.min(
                            feature_map[r, :, c])))

        #print(torch.max(feature_map))
        feature_map = feature_map.permute(1, 0, 2)

        feature_map = Image.fromarray(np.uint8(feature_map))

        feature_map = transform(feature_map)
        #print(torch.max(feature_map))
        data = Variable(feature_map).float().to(device=device)
        data=data.unsqueeze(dim=0)
        STMap = data[:, :, :, 0:256]

        rPPGNet_name='rPPGNetResizew36n256fn5fi1_new'
        rPPGNet=model.rPPGNet()
        rPPGNet = torch.load(rPPGNet_name, map_location=device)
        #print('load ' + rPPGNet_name + ' right')
        rPPGNet.to(device=device)
        Wave_pr = rPPGNet(STMap)

        # Move the tensor back to CPU before converting to numpy
        Wave_pr = Wave_pr.cpu().detach().numpy()
        Wave_pr = np.squeeze(Wave_pr)
        Wave_pr=(Wave_pr-np.min(Wave_pr))/(np.max(Wave_pr)-np.min(Wave_pr))
        #print(Wave_pr.shape)
        final_wave_pr=np.concatenate((final_wave_pr,Wave_pr))

        with open(ground_truth_path) as file:
            bvp = np.array(file.read().split(), dtype=float)                #strip().split('\n')
        bvp=bvp[i*256:i*256+256]
        bvp =(bvp - np.min(bvp)) / (np.max(bvp) - np.min(bvp))
        bvp = bvp.astype('float32')
        final_bvp_gt=np.concatenate((final_bvp_gt,bvp))
    plt.plot(final_wave_pr[256:256+256], label='Predicted Wave')
    plt.plot(final_bvp_gt[256:256+256], label='Ground Truth Wave')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('rPPG Waveform Prediction')
    plt.legend()
    plt.show()
    plt.savefig('graph_gt_predicted.png')

test(feature_map_path,ground_truth_path)