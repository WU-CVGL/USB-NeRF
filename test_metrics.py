from diff_img import *
import os
import torch
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import math
from lpips import lpips
import cv2
import run_nerf_helpers

os.environ["KMP_BLOCKTIME"] = "0"
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

new_save_dir = './baseline'


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def SSIM(img1, img2):
    (_, channel, _, _) = img1.size()
    window_size = 11
    window = create_window(window_size, channel).cuda()
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def PSNR(img1, img2, mask=None):
    if mask is not None:
        mask = mask.cuda()
        mse = (img1 - img2) ** 2
        B,C,H,W=mse.size()
        mse = torch.sum(mse * mask.float()) / (torch.sum(mask.float())*C)
    else:
        mse = torch.mean( (img1 - img2) ** 2 )
    
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def test():
    RESULTS_ALL_DICT = {}
    RESULTS_DIR = '../rolling-shutter-video/logs/Unreal-RS'
    DATASET = sorted(os.listdir(RESULTS_DIR))
    METHOD_LIST = ["Linear", "Cubic"]

    for datasets_name in DATASET:
        RESULTS_ALL_DICT[datasets_name+"_PSNR"] = {}
        RESULTS_ALL_DICT[datasets_name+"_SSIM"] = {}
        RESULTS_ALL_DICT[datasets_name+"_LPIPS"] = {}

        for method in METHOD_LIST:

            imgs_sharp_dir = os.path.join('../rolling-shutter-video/data/Unreal-RS', datasets_name, 'mid')
            imgs_render_dir = os.path.join(RESULTS_DIR, datasets_name, datasets_name+'_'+method, 'test_poses_mid', 'img_test_200000')
            save_dir = os.path.join(new_save_dir, method, datasets_name)
            os.makedirs(save_dir, exist_ok=True)
            
            imgs_sharp = run_nerf_helpers.load_imgs(imgs_sharp_dir)
            imgs_render = run_nerf_helpers.load_imgs(imgs_render_dir)
            
            f_metric_all = open(save_dir + '/metric_all.txt', 'w')
            f_metric_avg = open(save_dir + '/metric_avg.txt', 'w')

            f_metric_all.write(
                    '# frame_id, PSNR_pred, PSNR_pred_mask, SSIM_pred, LPIPS_pred\n')
            f_metric_avg.write(
                    '# avg_PSNR_pred, avg_PSNR_pred_mask, avg_SSIM_pred, avg_LPIPS_pred\n')

            loss_fn_alex = lpips.LPIPS(net='alex')

            sum_psnr = 0.
            sum_psnr_mask = 0.
            sum_ssim = 0.
            sum_lpips = 0.
            sum_time = 0.
            n_frames = 0


            for i in range(imgs_render.shape[0]):
                # compute metrics
                predict_GS = imgs_render[i].permute(2, 0, 1).unsqueeze(0)
                GT_GS = imgs_sharp[i].permute(2, 0, 1).unsqueeze(0)

                psnr_pred = PSNR(predict_GS, GT_GS)
                ssim_pred = SSIM(predict_GS, GT_GS)

                # lpips_pred = 0.
                lpips_pred = loss_fn_alex(predict_GS, GT_GS)  # compute LPIPS

                sum_psnr += psnr_pred
                sum_ssim += ssim_pred
                sum_lpips += lpips_pred
                n_frames += 1

                print('PSNR(%.8f dB) SSIM(%.8f) LPIPS(%.8f)\n' %
                        (psnr_pred, ssim_pred, lpips_pred))
                f_metric_all.write('%d %.8f %.8f %.8f\n' % (
                        i, psnr_pred, ssim_pred, lpips_pred))

                psnr_avg = sum_psnr / n_frames
                psnr_avg_mask = sum_psnr_mask / n_frames
                ssim_avg = sum_ssim / n_frames
                lpips_avg = sum_lpips / n_frames

            print('PSNR_avg (%.6f dB) SSIM_avg (%.6f) LPIPS_avg (%.6f) ' % (
                    psnr_avg, ssim_avg, lpips_avg))
            f_metric_avg.write('%.6f %.6f %.6f\n' %
                                (psnr_avg, ssim_avg, lpips_avg))
            
            metrics = np.array([float(psnr_avg), float(ssim_avg), float(lpips_avg.squeeze())])
            
            RESULTS_ALL_DICT[datasets_name+"_PSNR"][method] = metrics[0]
            RESULTS_ALL_DICT[datasets_name+"_SSIM"][method] = metrics[1]
            RESULTS_ALL_DICT[datasets_name+"_LPIPS"][method] = metrics[2]
        
    import pandas as pd
    pd.DataFrame(RESULTS_ALL_DICT).to_csv(os.path.join('./baseline', 'USB_NeRF.csv'))

if __name__=='__main__':
    test()