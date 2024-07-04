from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math



class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, sample):
        video_x, video_y, clip_average_HR, ecg_label, frame_rate, scale= sample['video_x'], sample['video_y'], sample['clip_average_HR'], sample['ecg'], sample['frame_rate'], sample['scale']
        new_video_x = (video_x - 127.5) / 128
        new_video_y = (video_y - 127.5) / 128
        return {'video_x': new_video_x, 'video_y': new_video_y, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        # print(2)
        video_x, video_y, clip_average_HR, ecg_label, frame_rate, scale= sample['video_x'], sample['video_y'], sample['clip_average_HR'], sample['ecg'], sample['frame_rate'], sample['scale']

        h, w = video_x.shape[1], video_x.shape[2]
        h1, w1 = video_y.shape[1], video_y.shape[2]
        new_video_x = np.zeros((video_x.shape[0], h, w, 3))
        new_video_y = np.zeros((video_y.shape[0], h1, w1, 3))
        p = random.random()
        if p < 0.5:
            # print('Flip')
            for i in range(video_x.shape[0]):
                # video
                image = video_x[i, :, :, :]
                image = cv2.flip(image, 1)
                new_video_x[i, :, :, :] = image
                image1 = video_y[i, :, :, :]
                image1 = cv2.flip(image1, 1)
                new_video_y[i, :, :, :] = image1
            return {'video_x': new_video_x, 'video_y': new_video_y, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}

        else:
            # print('no Flip')
            return {'video_x': video_x, 'video_y': video_y, 'clip_average_HR': clip_average_HR, 'ecg': ecg_label, 'frame_rate': frame_rate, 'scale':scale}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        # video_x, video_y, clip_average_HR, ecg_label, frame_rate, scale= sample['video_x'], sample['video_y'], sample['clip_average_HR'], sample['ecg'], sample['frame_rate'], sample['scale']
        video_y = sample['video_y']
        video_x = sample['video_x']

        clip_average_HR = sample['clip_average_HR']
        ecg_label = sample['ecg']
        frame_rate = sample['frame_rate']
        scale= sample['scale']

        video_x = video_x.transpose((3, 0, 1, 2))
        video_x = np.array(video_x)
        video_y = video_y.transpose((3, 0, 1, 2))
        video_y = np.array(video_y)
        clip_average_HR = np.array(clip_average_HR)

        frame_rate = np.array(frame_rate)

        ecg_label = np.array(ecg_label)

        scale = np.array(scale)

        return {'video_x': torch.from_numpy(video_x.astype(float)).float(),
                'video_y': torch.from_numpy(video_y.astype(float)).float(),
                'clip_average_HR': torch.from_numpy(clip_average_HR.astype(float)).float(),
                'ecg': torch.from_numpy(ecg_label.astype(float)).float(),
                'frame_rate': torch.from_numpy(frame_rate.astype(float)).float(),
                'scale':torch.from_numpy(scale.astype(float)).float()}


# train
class VIPL_train(Dataset):
    """MAHNOB  video +  Seg_labels  """

    def __init__(self, scale,frames ,test=False, transform=None):

        # self.test = test
        self.parent="/scratch/siddharths.scee.iitmandi/DATASET-2/"
        subject_list = os.listdir(self.parent)
        # subject_list.remove('subject11')
        # subject_list.remove('subject18')
        # subject_list.remove('subject20')
        # subject_list.remove('subject24')
        subject_list.sort()
        subject_list=[i for i in subject_list if os.path.exists(self.parent+i+"/frames")]
        self.vdPath_list = subject_list
        length=len(self.vdPath_list)
        if test:
            self.vdPath_list = self.vdPath_list[:int(length*0.2)]
            # self.vdPath_list = ["subject1","subject3","subject9"]
        else:
            self.vdPath_list = self.vdPath_list[int(length*0.2):]
        # if not test:
        #     self.parent = "/home/siddharths.scee.iitmandi/Home/DATASET-2/"
        #     self.vdPath_list=["subject3","subject32","subject35","subject36","subject37","subject41","subject44","subject45","subject48","subject49"]
        # else:
        #     self.vdPath_list=["subject1","subject9","subject11","subject12","subject13"]
        #     self.parent = "/scratch/siddharths.scee.iitmandi/DATASET-2/"
            # [self.vdPath_list.remove(i) for i in ["subject3","subject32","subject35","subject36","subject37","subject41","subject44","subject45","subject48","subject49"]]
        # if test:
        #     self.vdPath_list = subject_list[30:]
        # else:
        #     self.vdPath_list = subject_list[:30]
        self.transform = transform
        self.scale = scale
        self.frames = frames
        self.segment_numbers = math.floor(1367/frames)
        self.frame_rate = []
        self.clhr = []
        self.Trace = [[]]
        self.idx_scale = 0
        self.idx_scale_vice = 0
        
        for i in range(len(self.vdPath_list)):
            video_path = self.parent + self.vdPath_list[i] + "/vid.avi"
            capture = cv2.VideoCapture(video_path)
            self.frame_rate.append(capture.get(cv2.CAP_PROP_FPS))
            path = self.parent + self.vdPath_list[i] + "/ground_truth.txt"

            f = open(path)
            data = f.readlines()
            clhr_x = list(data[1].split())
            clhr_x = [str.replace('e', 'E') for str in clhr_x]
            self.clhr.append(clhr_x)


            data = list(data[0].split())
            data = [str.replace('e', 'E') for str in data]
            self.Trace.append([])
            for j in range(len(data)):
                self.Trace[i].append(float(data[j]))


    def __len__(self):
        return len(self.vdPath_list) * self.segment_numbers

    def __getitem__(self, idx):
        clip = idx % self.segment_numbers

        idx = int(idx / self.segment_numbers) 
        start_frame = self.frames * clip




        sumHR = 0.0
        for kj in range(start_frame, start_frame + self.frames):
            sumHR += float(self.clhr[idx][kj])
        clip_average_HR = sumHR / self.frames
        if (clip_average_HR <= 40):

             clip_average_HR = 90.0


        # video_x = self.get_single_video_x("/" + self.vdPath_list[idx] + '/SegFacePic/' + "{}/".format(self.scale[self.idx_scale]), start_frame + 1)
        # video_y = self.get_single_video_x("/" + self.vdPath_list[idx] + '/SegFacePic/' + "{}/".format(self.scale[self.idx_scale_vice]), start_frame + 1)
        video_x = self.get_single_video_x(self.parent + self.vdPath_list[idx] + '/frames/' , start_frame + 1)
        video_y = self.get_single_video_x(self.parent + self.vdPath_list[idx] + '/frames/' , start_frame + 1)


        ecg_label = self.Trace[idx][start_frame:start_frame + self.frames]

        sample = {'video_x': video_x,'video_y': video_y, 'frame_rate': self.frame_rate[idx], 'ecg': ecg_label, 'clip_average_HR': clip_average_HR, 'scale':self.scale[self.idx_scale]}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_video_x(self, video_jpgs_path, start_frame):
        image_name ='frame-' + '1' + '.png'
        image_path = os.path.join(video_jpgs_path, image_name)
        # print(image_path)
        image_shape = cv2.imread(image_path).shape
        video_x = np.zeros((self.frames, image_shape[0], image_shape[1], 3))

        # image_id = start_frame
        for i in range(self.frames):
            s = start_frame + i
            image_name = 'frame-' + str(s) + '.png'

            # face video
            image_path = os.path.join(video_jpgs_path, image_name)

            tmp_image = cv2.imread(image_path)

            if tmp_image is None:  # It seems some frames missing

                tmp_image = cv2.imread('./_1.jpg')
                print("______________________.jpg")


            video_x[i, :, :, :] = tmp_image

        return video_x

    def set_scale(self, idx_scale,idx_scale_vice):
        self.idx_scale = idx_scale
        self.idx_scale_vice = idx_scale_vice
