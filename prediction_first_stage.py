import numpy as np
import torch.nn as nn
import os,shutil,torch
import matplotlib.pyplot as plt
from utils_1.config import opt
from load_data import IMG_Folder
from model_1 import ScaleDense
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_absolute_error

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def metric(output, target):
    target = target.data.numpy()
    pred = output.cpu()  
    pred = pred.data.numpy()
    mae = mean_absolute_error(target,pred)
    return mae

def main():
    # ======== define data loader and CUDA device ======== #
    test_data = IMG_Folder(opt.excel_path, opt.test_folder)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ========  build and set model  ======== #  
    if opt.model == 'ScaleDense':
        model = ScaleDense.ScaleDense(8, 5, opt.use_gender)
    else:
        print('Wrong model choose')




    # ======== load trained parameters ======== #
    model = nn.DataParallel(model).to(device)
    criterion = nn.MSELoss().to(device)
    model.load_state_dict(torch.load(os.path.join(opt.output_dir+opt.model_name))['state_dict'])

    # ======== build data loader ======== #
    test_loader = torch.utils.data.DataLoader(test_data
                                             ,batch_size=opt.batch_size
                                             ,num_workers=opt.num_workers
                                             ,pin_memory=True
                                             ,drop_last=True
                                             )

    # ======== test preformance ======== #
    test( valid_loader=test_loader
        , model=model
        , criterion=criterion
        , device=device
        , save_npy=True
        , npy_name=opt.npz_name
        , figure=True
        , figure_name=opt.plot_name)

def test(valid_loader, model, criterion, device
        , save_npy=False,npy_name='test_result.npz'
        , figure=False, figure_name='ScaleDense_best_model_1.png'):

    '''
    [Do Test process according pretrained model]

    Args:
        valid_loader (torch.dataloader): [test set dataloader defined in 'main']
        model (torch CNN model): [pre-trained CNN model, which is used for brain age estimation]
        criterion (torch loss): [loss function defined in 'main']
        device (torch device): [GPU]
        save_npy (bool, optional): [If choose to save predicted brain age in npy format]. Defaults to False.
        npy_name (str, optional): [If choose to save predicted brain age, what is the npy filename]. Defaults to 'test_result.npz'.
        figure (bool, optional): [If choose to plot and save scatter plot of predicted brain age]. Defaults to False.
        figure_name (str, optional): [If choose to save predicted brain age scatter plot, what is the png filename]. Defaults to 'True_age_and_predicted_age.png'.

    Returns:
        [float]: MAE and pearson correlation coeficent of predicted brain age in teset set.
    '''

    losses = AverageMeter()
    MAE = AverageMeter()

    model.eval() # switch to evaluate mode
    out, targ, ID = [], [], []

    target_numpy, predicted_numpy, ID_numpy = [], [], []
    gt_10, gt_25, gt_26 = [], [], []
    pr_10, pr_25, pr_26 = [], [], []
    tar_10, pred_10 = [], []
    tar_25, pred_25 = [], []
    tar_26, pred_26 = [], []
    print('======= start prediction =============')
    # ======= start test programmer ============= #
    with torch.no_grad():
        for _, (input, ids ,target,male) in enumerate(valid_loader):
            input = input.to(device).type(torch.FloatTensor)

            # print("input.shape:", input.shape)
            # ======= convert male lable to one hot type ======= #
            male = torch.unsqueeze(male,1)
            male = torch.zeros(male.shape[0],2).scatter_(1,male,1)
            male = male.type(torch.FloatTensor).to(device)

            target = torch.from_numpy(np.expand_dims(target,axis=1))
            target = target.type(torch.FloatTensor).to(device)

            # ======= compute output and loss ======= #
            if opt.model == 'ScaleDense' :
                output = model(input,male)

            else:
                output = model(input)
            out.append(output.cpu().numpy())
            targ.append(target.cpu().numpy())
            ID.append(ids)
            loss = criterion(output, target)
            mae = metric(output.detach(), target.detach().cpu())

            # ======= measure accuracy and record loss ======= #
            losses.update(loss, input.size(0))
            MAE.update(mae, input.size(0))

        targ = np.asarray(targ)
        out = np.asarray(out)
        ID = np.asarray(ID)

        for idx in targ:
            for i in idx:
                target_numpy.append(i)

        for idx in out:

            for i in idx:
                predicted_numpy.append(i)





        for idx in ID:
            for i in idx:
                ID_numpy.append(i)
        for i in range(len(target_numpy)):
            if target_numpy[i] < 10:
                tar_10.append(target_numpy[i])
                pred_10.append(predicted_numpy[i])
            elif target_numpy[i] < 25:

                tar_25.append(target_numpy[i])
                pred_25.append(predicted_numpy[i])
            else:
                tar_26.append(target_numpy[i])
                pred_26.append(predicted_numpy[i])


        for i in range(len(target_numpy)):
            if target_numpy[i] < 10:
                gt_10.append(1)
                gt_25.append(0)
                gt_26.append(0)
            elif target_numpy[i] < 25:
                gt_10.append(0)
                gt_25.append(1)
                gt_26.append(0)

            else:
                gt_10.append(0)
                gt_25.append(0)
                gt_26.append(1)


        for i in range(len(predicted_numpy)):
            if predicted_numpy[i] < 10:
                pr_10.append(1)
                pr_25.append(0)
                pr_26.append(0)
            elif predicted_numpy[i] < 25:
                pr_10.append(0)
                pr_25.append(1)
                pr_26.append(0)
            else:
                pr_10.append(0)
                pr_25.append(0)
                pr_26.append(1)

        gt_10, gt_25, gt_26, pr_10, pr_25, pr_26 = np.asarray(gt_10), np.asarray(gt_25), np.asarray(gt_26), np.asarray(pr_10), np.asarray(pr_25), np.asarray(pr_26)
        acc_10, acc_25, acc_26 = np.sum(gt_10==pr_10)/30, np.sum(gt_25==pr_25)/30, np.sum(gt_26==pr_26)/30
        precision_10, precision_25, precision_26 = np.sum(gt_10*pr_10)/(np.sum(gt_10+pr_10)-np.sum(gt_10*pr_10)), np.sum(gt_25*pr_25)/(np.sum(gt_25+pr_25)-np.sum(gt_25*pr_25)), np.sum(gt_26*pr_26)/(np.sum(gt_26+pr_26)-np.sum(gt_26*pr_26))
        rec_10, rec_25, rec_26 = np.sum(gt_10*pr_10)/np.sum(gt_10), np.sum(gt_25*pr_25)/np.sum(gt_25), np.sum(gt_26*pr_26)/np.sum(gt_26)
        f1_10, f1_25, f1_26 = 2*precision_10*rec_10/(precision_10+rec_10), 2*precision_25*rec_25/(precision_25+rec_25), 2*precision_26*rec_26/(precision_26+rec_26)
        print("acc_10, acc_25, acc_26, acc_avg:", acc_10, acc_25, acc_26, (acc_10+acc_25+acc_26)/3)
        print("precision_10, precision_25, precision_26, precision_avg:", precision_10, precision_25, precision_26, (precision_10+precision_25+precision_26)/3)
        print("rec_10, rec_25, rec_26, rec_avg:", rec_10, rec_25, rec_26, (rec_10+rec_25+rec_26)/3)
        print("f1_10, f1_25, f1_26, f1_avg:", f1_10, f1_25, f1_26, (f1_10+f1_25+f1_26)/3)





        target_numpy = np.asarray(target_numpy)
        predicted_numpy = np.asarray(predicted_numpy)

        print("target_numpy.shape, predicted_numpy.shape:", target_numpy.shape, predicted_numpy.shape)
        ID_numpy = np.asarray(ID_numpy)
        tar_10, tar_25, tar_26 = np.asarray(tar_10), np.asarray(tar_25), np.asarray(tar_26)
        pred_10, pred_25, pred_26 = np.asarray(pred_10), np.asarray(pred_25), np.asarray(pred_26)

        error_10, error_25, error_26 = pred_10-tar_10, pred_25-tar_25, pred_26-tar_26
        abs_10, abs_25, abs_26 = np.abs(error_10), np.abs(error_25), np.abs(error_26)


        errors = predicted_numpy - target_numpy
        abs_errors = np.abs(errors)

        errors = np.squeeze(errors,axis=1)
        abs_errors = np.squeeze(abs_errors,axis=1)
        target_numpy = np.squeeze(target_numpy,axis=1)
        predicted_numpy = np.squeeze(predicted_numpy,axis=1)


        # ======= output several results  ======= #
        print('===============================================================\n')
        print(
            'TEST  : [steps {0}], Loss {loss.avg:.4f},  MAE:  {MAE.avg:.4f} \n'.format(
            len(valid_loader), loss=losses, MAE=MAE))

        print('STD_err = ', np.std(errors))  
        print(' CC:    ',np.corrcoef(target_numpy,predicted_numpy))
        print('PAD spear man cc',spearmanr(errors,target_numpy,axis=1))
        print('spear man cc',spearmanr(predicted_numpy,target_numpy,axis=1))
        print('mean pad:',np.mean(errors))

        print('\n =================================================================')

        print("abs_errors.mean(), abs_10.mean(), abs_25.mean(), abs_26.mean():", abs_errors.mean(), abs_10.mean(), abs_25.mean(), abs_26.mean())
        if save_npy:
            savepath = os.path.join(opt.output_dir,npy_name)
            np.savez(savepath 
                    ,target=target_numpy
                    ,prediction=predicted_numpy
                    ,ID=ID_numpy)

        # ======= Draw scatter plot of predicted age against true age ======= #
        if figure is True:
            plt.figure()
            lx = np.arange(np.min(target_numpy),np.max(target_numpy))
            plt.plot(lx,lx,color='red',linestyle='--')
            plt.scatter(target_numpy,predicted_numpy)
            plt.xlabel('GT ICG R15')
            plt.ylabel('Predicted ICG R15')
            # plt.show()
            plt.savefig(opt.output_dir+figure_name)
        return MAE ,np.corrcoef(target_numpy,predicted_numpy)


if __name__ == "__main__":
    main()
