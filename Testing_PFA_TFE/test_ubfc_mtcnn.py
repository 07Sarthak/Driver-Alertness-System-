import torch
import matplotlib.pyplot as plt
import argparse, os
import cv2
import numpy as np


from torchvision import transforms
from model.PhysNet_PFE_TFA_crcloss import PhysNet_padding_ED_peak
from dataloader.dataloader import MHDataLoader
from dataloader.LoadVideotrain_ubfc_mtcnn import VIPL_train, Normaliztion, ToTensor, RandomHorizontalFlip

import torch.nn as nn
from utils.TorchLossComputer import TorchLossComputer


def FeatureMap2Heatmap(x, feature1, feature2):
    ## initial images
    ## initial images
    x = x.repeat(2,1,1,1,1)
    org_img = x[0, :, 32, :, :].cpu()
    org_img = org_img.data.numpy() * 128 + 127.5
    org_img = org_img.transpose((1, 2, 0))


    cv2.imwrite(args.log + '/mtcnn/' + args.log + '_x_visual.jpg', org_img)

    ## first feature
    feature_first_frame = feature1[0, :, 16, :, :].cpu()  ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    heatmap = np.asarray(heatmap, dtype=np.uint8)

    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # COLORMAP_WINTER     COLORMAP_JET
    heat_img = cv2.resize(heat_img, (128, 128))
    cv2.imwrite(args.log + '/mtcnn/' + args.log + '_x_heatmap1.jpg', heat_img)

    ## second feature
    feature_first_frame = feature2[0, :, 8, :, :].cpu()  ## the middle frame
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i, :, :], 2).view(feature_first_frame.size(1),
                                                                   feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()

    heatmap = np.asarray(heatmap, dtype=np.uint8)

    heat_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # COLORMAP_WINTER     COLORMAP_JET
    heat_img = cv2.resize(heat_img, (128, 128))
    cv2.imwrite(args.log + '/mtcnn/' + args.log + '_x_heatmap2.jpg', heat_img)


class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson, self).__init__()
        return

    def forward(self, preds, labels):  # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))


            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def test(condition, scale):
    condition = args.version
    device_ids = args.gpu
    frames = args.frames
    isExists = os.path.exists(args.log + '/mtcnn/' + 'train_version' + str(condition) + '_' +str(args.epochs))
    if not isExists:
        os.makedirs(args.log+ '/mtcnn/' + 'train_version' + str(condition)+ '_' +str(args.epochs))
    log_file = open(args.log + '/mtcnn/' + 'train_version' + str(condition) +  '_' +str(args.epochs) + '/'  +args.log + '_test_condition_' + 'scale'+ str(scale) + '_.txt', 'w')
    log_file.write("mtcnn_resized\n\n")
    print("mtcnn_resized\n\n")

    for ik in range(0, 1):

        index = ik + 1

        log_file.write('cross-valid : %d' % (index))
        log_file.write("\n")
        log_file.flush()

        finetune = args.finetune

        print('test!\n')
        log_file.write('test!\n')
        log_file.flush()

        model = PhysNet_padding_ED_peak(frames = frames, device_ids = device_ids, hidden_layer = args.hidden_layer)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda(device=device_ids)
        map_location = 'cuda:' + str(device_ids)
        model.load_state_dict(
            torch.load(args.log + '/mtcnn/weights/' + args.log + '_con_' + str(condition) + '_%d_%d.pkl' % (1, args.epochs), map_location=map_location))

        torch.no_grad()

        criterion_Pearson = Neg_Pearson()
        scale = args.scale
        loss_rPPG_avg = AvgrageMeter()
        loss_peak_avg = AvgrageMeter()
        loss_hr_rmse = AvgrageMeter()

        model.eval()
        PURE_trainDL = VIPL_train(scale,frames,transform=transforms.Compose([Normaliztion(), RandomHorizontalFlip(), ToTensor()]),
                                  test=True)

        dataloader_train = MHDataLoader(args,PURE_trainDL, batch_size=1, shuffle=False, pin_memory=not args.cpu)
        with torch.no_grad():
            for i, sample_batched in enumerate(dataloader_train):
                inputs_1, ecg = sample_batched['video_x'].cuda(device=device_ids), sample_batched['ecg'].cuda(device=device_ids)
                inputs_2 = sample_batched['video_y'].cuda(device=device_ids)
                frame_rate = sample_batched['frame_rate'].cuda(device=device_ids)
                ecg = ecg.repeat(2,1)
                # clip_average_HR = clip_average_HR.repeat(2,1)
                frame_rate = frame_rate.repeat(2,1)

                rPPG_peak, x_visual, x_visual3232, x_visual1616 = model(inputs_1,inputs_2)
                rPPG = rPPG_peak[:, 0, :]

                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize2
                ecg = (ecg - torch.mean(ecg)) / torch.std(ecg) 

                if i % args.echo_batches == args.echo_batches - 1:  # print every 50 mini-batches
                    try:
                        FeatureMap2Heatmap(x_visual,x_visual3232,x_visual1616)
                    except Exception as error:
                        print(f"An error occured in minibatch {i+1}: \n",error)

                for t in range(0, int(len(rPPG.cpu().detach().numpy().tolist())/2)):
                    draw_scatter(ecg.cpu().detach().numpy().tolist()[t], rPPG.cpu().detach().numpy().tolist()[t], frames,
                                 str(i) + "_" + str(t), condition, scale)
                    log_file.write("-----------------------------------------------------------------------\n")
                    log_file.write("frame_rate:\n")
                    log_file.write(str(frame_rate.cpu().detach().numpy().tolist()[t]))
                    log_file.write("\n")
                    # log_file.write("clip_average_HR:\n")
                    # log_file.write(str(clip_average_HR.cpu().detach().numpy().tolist()[t]))
                    log_file.write("\n")
                    log_file.write("ecg:\n")
                    log_file.write(str(ecg.cpu().detach().numpy().tolist()[t]))
                    log_file.write("\n")
                    log_file.write("pre:\n")
                    log_file.write(str(rPPG.cpu().detach().numpy().tolist()[t]))
                    log_file.write("\n")
                    print("i:", i, "t:", t)

                loss_rPPG = criterion_Pearson(rPPG, ecg)

                # clip_average_HR = (clip_average_HR - 40)  # [40, 180]
                # fre_loss = 0.0
                # train_rmse = 0.0

                # for bb in range(inputs_1.shape[0]+inputs_2.shape[0]):
                #     fre_loss_temp, train_rmse_temp = TorchLossComputer.cross_entropy_power_spectrum_forward_pred(rPPG[bb],frame_rate[bb],device_ids)
                #     fre_loss = fre_loss + fre_loss_temp
                #     train_rmse = train_rmse + train_rmse_temp
                # fre_loss = fre_loss / (inputs_1.shape[0]+inputs_2.shape[0])
                # train_rmse = train_rmse / (inputs_1.shape[0]+inputs_2.shape[0])

                # n = inputs_1.size(0)+inputs_2.size(0)
                # loss_rPPG_avg.update(loss_rPPG, n)
                # loss_peak_avg.update(fre_loss, n)
                # loss_hr_rmse.update(train_rmse, n)

                log_file.write("\n")
                log_file.write("\n")
                log_file.flush()

    print('Finished Test')
    # print("rPPG Loss: ",loss_rPPG_avg.avg)
    # print("peak Loss: ",loss_peak_avg.avg)
    # print("HR rmse: ",loss_hr_rmse.avg)
    log_file.close()


def draw_scatter(GT, Pre, n, s, condition, scale):
    x1 = range(n)
    y1 = GT
    y2 = Pre

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title('Result Analysis')
    ax1.set_xlabel('time')
    ax1.set_ylabel('rPPG')
    ax1.plot(x1, y1, c='r', ls='-')  
    ax1.plot(x1, y2, c='blue', ls='-')
    isExists = os.path.exists("./" + args.log + "/mtcnn/" + "bluePre_redGT_" + str(condition) +'_' + str(args.epochs) + "/" + 'scale' +str(scale))
    if not isExists:
        os.makedirs("./" + args.log + "/mtcnn/" + "bluePre_redGT_" + str(condition) +'_' + str(args.epochs) + "/" + 'scale' +str(scale))

    plt.savefig("./" + args.log + "/mtcnn/" +"bluePre_redGT_" + str(condition) +'_' + str(args.epochs) + "/" + 'scale' +str(scale) + '/' + s + ".png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=str, default=1, help='the gpu id used for predict')
    parser.add_argument('--frames',type=int,default=160,help='how many frames')
    parser.add_argument('--hidden_layer',type=int,default=128,help='how many point in hidden')    
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  # default=0.0001
    parser.add_argument('--step_size', type=int, default=50,
                        help='stepsize of optim.lr_scheduler.StepLR, how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 200
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="SSTTFinallog_Constrative", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--test', default=True, help='whether test')
    parser.add_argument('--version', default=3, help='version info')
    parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
    parser.add_argument('--scale', type=str, default='', help='super resolution scale')
    parser.add_argument('--cpu', action='store_true',help='use cpu only')
    args = parser.parse_args()
    if args.scale=='':
        args.scale = [1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
    else:
        args.scale = list(map(lambda x: float(x), args.scale.split('+')))
    backup = args.scale 
    if args.gpu=='':
        args.gpu = 0
    '''else:
        number = args.gpu.split(',')
        args.gpu = [int(x) for x in number] '''
    for scale in backup:
        args.scale = [scale]
        test(args.version, scale)

