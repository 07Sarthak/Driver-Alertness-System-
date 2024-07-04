import time 
begin=time.time()

from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from extract import extract_face
import os
import torch

torch.cuda.empty_cache()

def pad_image(image, target_size):
    """
    Pad the image to match the target size.
    """
    height, width = image.shape[:2]
    pad_height = target_size[0] - height
    pad_width = target_size[1] - width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
mtcnn = MTCNN(select_largest=True, device=device)
path="/scratch/siddharths.scee.iitmandi/collected_data/"
subs=os.listdir(path)
# subs=[i for i in subs if not os.path.exists(path+i+"/padded")]
print(subs)
for sub in subs:
    print(sub)
    save=path+sub+"/mtcnn_resized_collected_data/"
    if not os.path.exists(save):
        os.mkdir(save)

    v_cap = cv2.VideoCapture(path+sub+"/vid.avi")
    success, frame = v_cap.read()
    # w,h=(0,0)
    # while success:
    #     boxes, probs = mtcnn.detect(frame)
    #     box=boxes[0]
    #     box=[int(x) for x in box.tolist()]
    #     w=max(w,box[2]-box[0])
    #     h=max(h,box[3]-box[1])
    #     success, frame = v_cap.read()
    
    # for filename in os.listdir(path+sub+"/mtcnn/"):
    #     img = cv2.imread(os.path.join(path+sub+"/mtcnn/", filename))
    #     height, width = img.shape[:2]
    #     h = max(h, height)
    #     w = max(w, width)
    
    # print(w,h)

    def extract_integer(filename):                      
        return int(filename.split('.')[0])
    # v_cap = cv2.VideoCapture(path+sub+"/vid.avi")
    i=0
    # success, frame = v_cap.read()
    # ims=os.listdir(path+sub+"/mtcnn/")
    # ims=sorted(ims,key=extract_integer)
    # for filename in ims:
    #     out = cv2.imread(os.path.join(path+sub+"/mtcnn/", filename))
    #     try:
    #         padded=pad_image(out, (h,w))
    #     except:
    #         print(sub,i,f": max h,w ={(h,w)} , im_size= {out.shape[:2]}")
    #     cv2.imwrite(save + str(i) + ".png",padded)
    #     if i%100==0:
    #         print(i)
        # i+=1

    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(frame)
        box=boxes[0]
        box=[int(x) for x in box.tolist()]
        # blank_img=np.zeros((h,w))
        # blank_img[:box[3]-box[1], :box[2]-box[0]]=frame[box[1]:box[3], box[0]:box[2]]
        # cv2.imwrite(save + str(i) + ".png",blank_img)
        # # face=extract_face(frame, box)
        # padded=frame[box[1]:box[3], box[0]:box[2]]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = frame[box[1]:box[3], box[0]:box[2]]
        out=cv2.resize(
            img,
            (64, 64),
            interpolation=cv2.INTER_AREA
        ).copy()
        # try:
        #     padded=pad_image(out, (h,w))
        # except:
        #     print(sub,i,f": max h,w ={(h,w)} , im_size= {out.shape[:2]}")
        cv2.imwrite(save + str(i) + ".png",out)
        if i%100==0:
            print(i)
        i+=1
        success, frame = v_cap.read()
end=time.time()
print(f"Total runtime of the program is {end - begin}") 
