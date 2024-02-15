import os
import cv2
import torch

from nerf import *
import optimize_pose
import compose
import torchvision.transforms.functional as torchvision_F

import matplotlib.pyplot as plt

from tvloss import EdgeAwareVariationLoss, GrayEdgeAwareVariationLoss
from metrics import compute_img_metric


def train(args):
    print('args.barf: ', args.barf, 'args.barf_start_end: ', args.barf_start, args.barf_end)
    print('tv_width: ', args.tv_width_nerf)
    print('lambda: ', args.tv_loss_lambda)

    args.readout_time = args.readout_time * args.factor

    print('The period between two frames is:', args.period)
    print('The row time difference is:', args.readout_time)

    optimize_se3 = args.optimize_se3
    optimize_nerf = args.optimize_nerf
    
    args.xys_remap = None
    K = None

    if args.dataset_type == 'llff':
        images, poses_init, bds_start, render_poses = load_llff_data(args.datadir, factor=args.factor, recenter=True,
                                                                      bd_factor=.75, spherify=args.spherify, focal = args.focal)
        
        images_gs_f, images_gs_m = load_gs_data(args)
        
        K = np.array([
            [548.409,   0,          384], 
            [0,         548.409,    240], 
            [0,         0,          1]])
        
        K[:2, :3]=K[:2, :3] / args.factor

    
    elif args.dataset_type == 'tum-rs':
        images, poses_init, bds_start, render_poses = load_llff_TUM.load_llff_data(args.datadir, factor=args.factor, recenter=True,
                                                                      bd_factor=.75, spherify=args.spherify, focal = args.focal)
        images_gs_f, images_gs_m = None, None
        
        K = np.array([
            [739.1654756101043,     0,                  625.826167006398], 
            [0,                     739.1438452683457,  517.3370973594253], 
            [0,                     0,                  1]])

        K_ = np.array(K)
        K_[:2, :3] = K_[:2, :3] / args.factor
        
        
        dist_coeffs = np.array([0.019327620961435945, 0.006784242994724914, -0.008658628531456217, 0.0051893686731546585])

        H, W = images.shape[1], images.shape[2]

        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        xys = np.stack((xs, ys), axis=-1) # (H, W, 2)
        xys = xys * args.factor
        args.xys_remap = cv2.fisheye.undistortPoints(xys.astype(np.float32), K, dist_coeffs, R=np.eye(3), P=K_)

        K = K_

    elif args.dataset_type == 'gopro':
        images, poses_init, bds_start, render_poses = load_llff_data(args.datadir,
                                                                      factor=args.factor, recenter=True,
                                                                      bd_factor=.75, spherify=args.spherify, focal = args.focal)
        images_gs_f, images_gs_m = None, None
            
    else:
        print('Unknown dataset type', args.dataset_type)
        return
    
    assert (poses_init.shape[0]==images.shape[0])
    hwf = poses_init[0, :3, -1]

    # split train/val/test
    i_test = torch.tensor([poses_init.shape[0], poses_init.shape[0]+1, poses_init.shape[0]+2]).long()
    i_val = i_test
    i_train = torch.Tensor([i for i in torch.arange(int(images.shape[0])) if
                                (i not in i_test and i not in i_val)]).long()

    # transfer camera poses to se3
    init_se3 = SE3_to_se3_N(poses_init[:, :3, :4])

    print('Loaded llff', images.shape, hwf, args.datadir)

    print('DEFINING BOUNDS')
    if args.no_ndc:
        near = torch.min(bds_start) * .9
        far = torch.max(bds_start) * 1.

    else:
        near = 0.
        far = 1.
        print('NEAR FAR', near, far)

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = torch.Tensor([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    print('camera intrinsic parameters: \n', K, '!')

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
        
    model = optimize_pose.Model(init_se3)
    graph = model.build_network(args)
    optimizer, optimizer_se3 = model.setup_optimizer(args)

    if args.load_weights:
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(args.weight_iter))
        graph_ckpt = torch.load(path)

        # only load paramaters of NeRF
        if args.only_optimize_SE3:
            # only load nerf and nerf_fine network
            delete_key = []
            for key, value in graph_ckpt['graph'].items():
                if key[:4] == 'se3.':
                    delete_key.append(key)

            pretrained_dict = {k: v for k, v in graph_ckpt['graph'].items() if k not in delete_key}
            print('only load nerf and nerf_fine network!!!!!')
            graph.load_state_dict(pretrained_dict, strict=False)
            global_step=graph_ckpt['global_step']
        # load full model
        else:
            graph.load_state_dict(graph_ckpt['graph'])
            optimizer.load_state_dict(graph_ckpt['optimizer'])
            optimizer_se3.load_state_dict(graph_ckpt['optimizer_se3'])
            if args.two_phase:
                global_step = 1
            else:
                global_step=graph_ckpt['global_step']

        print('Model Load Done!')
    else:
        print('Not Load Model!')


    N_iters = args.max_iter + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = 0
    if not args.load_weights:
        global_step = 0
    start = start + global_step

    # show image
    images_num = images.shape[0]
    # plt.show()
    if args.tv_loss:
        tvloss_fn_rgb = EdgeAwareVariationLoss(in1_nc=3)
        tvloss_fn_gray = GrayEdgeAwareVariationLoss(in1_nc=1, in2_nc=3)

    print('Number is {} !!!'.format(images_num))
    for i in trange(start, N_iters):
    ### core optimization loop ###
        if i == 0:
            init_nerf(graph.nerf)
            init_nerf(graph.nerf_fine)
        
        # used for testing
        # if args.render_rolling_shutter:
        #     with torch.no_grad():
        #         rolling_shutter_poses = graph.forward(i,   images_num, H, W, K, args)
        #         render_rolling_shutter_(i, graph, rolling_shutter_poses, H, W, K, args, dir='rolling_shutter', need_depth=False)


        if i % args.i_video == 0 and i > 0:
            ret, ray_idx, test_poses_mid, test_poses_start, test_poses_end, test_render_poses = graph.forward(i, images_num, H, W, K, args)
        elif i % args.i_img == 0 and i > 0:
            ret, ray_idx, test_poses_mid, test_poses_start, test_poses_end = graph.forward(i, images_num, H, W, K, args)
        else:
            ret, ray_idx = graph.forward(i,   images_num, H, W, K, args)
            
        # get image ground truth
        ray_idx = ray_idx.cpu().numpy()
        target_s = images[ray_idx[:,0], ray_idx[:,1], ray_idx[:,2], :]  # [N_, N_rand//N, 3]
        target_s = torch.tensor(target_s)

        if args.tv_loss and i >= args.n_tvloss:
            # if args.tv_loss_rgb:
            rgb_sharp_tv = ret['rgb_map'][-args.tv_width_nerf ** 2:]
            rgb_sharp_tv = rgb_sharp_tv.reshape(-1, args.tv_width_nerf, args.tv_width_nerf, 3)  # [NHWC]
            rgb_sharp_tv = torch.permute(rgb_sharp_tv, (0, 3, 1, 2))
            if args.tv_loss_gray:
                gray = ret['disp_map'][-args.tv_width_nerf ** 2:]
                depth_tv = gray
                depth_tv = depth_tv.reshape(-1, args.tv_width_nerf, args.tv_width_nerf, 1)  # [NHWC]
                depth_tv = torch.permute(depth_tv, (0, 3, 1, 2))

        # backward
        optimizer_se3.zero_grad()
        optimizer.zero_grad()

        if args.tv_loss is False:
            img_loss = img2mse(ret['rgb_map'], target_s)
        else:
            img_loss = img2mse(ret['rgb_map'][:-args.tv_width_nerf ** 2], target_s)

        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in ret:
            if args.tv_loss is False:
                img_loss0 = img2mse(ret['rgb0'], target_s)
            else:
                img_loss0 = img2mse(ret['rgb0'][:-args.tv_width_nerf ** 2], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if args.tv_loss and i >= args.n_tvloss:
            if args.tv_loss_rgb and args.tv_loss_gray:
                loss_tv_rgb = tvloss_fn_rgb(rgb_sharp_tv, mean=True)
                loss_tv_depth = tvloss_fn_gray(depth_tv, rgb_sharp_tv, mean=True)
                loss_tv = loss_tv_rgb + loss_tv_depth
            elif args.tv_loss_rgb:
                loss_tv = tvloss_fn_rgb(rgb_sharp_tv, mean=True)
            else:
                loss_tv = tvloss_fn_gray(depth_tv, rgb_sharp_tv, mean=True)

            loss = loss + args.tv_loss_lambda*loss_tv

        loss.backward()

        if optimize_nerf:
            optimizer.step()
        if optimize_se3:
            optimizer_se3.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = args.decay_rate
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        decay_rate_pose = args.decay_rate_pose
        new_lrate_pose = args.pose_lrate * (decay_rate_pose ** (i / decay_steps))
        for param_group in optimizer_se3.param_groups:
            param_group['lr'] = new_lrate_pose
        ###############################

        if i % args.i_print == 0:
            if args.tv_loss and i >= args.n_tvloss:
                tqdm.write(
                    f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss0.item()}, PSNR: {psnr.item()} TV Loss: {loss_tv.item()}")
            else:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  coarse_loss:, {img_loss0.item()}, PSNR: {psnr.item()}")

        if i < 10:
            print('coarse_loss:', img_loss0)

        if i % args.i_weights == 0 and i > 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': i,
                'graph': graph.state_dict(),
                'optimizer': optimizer.state_dict(),
                'optimizer_se3': optimizer_se3.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # render images
        if i % args.i_img == 0 and i > 0:
            args.train_state == 'val'
            with torch.no_grad():
                imgs_render_mid = render_image_test(i, graph, test_poses_mid, H//args.render_downsample, W//args.render_downsample, K/args.render_downsample, args, dir='test_poses_mid', need_depth=False)
                imgs_render_start = render_image_test(i, graph, test_poses_start, H//args.render_downsample, W//args.render_downsample, K/args.render_downsample, args, dir='test_poses_start', need_depth=False)
                # imgs_render_end = render_image_test(i, graph, test_poses_end, H, W, K, args,  dir='test_poses_end', need_depth=False)

            if images_gs_m != None:
                mse_render = compute_img_metric(images_gs_m, imgs_render_mid, 'mse')
                psnr_render = compute_img_metric(images_gs_m, imgs_render_mid, 'psnr')
                ssim_render = compute_img_metric(images_gs_m, imgs_render_mid, 'ssim')
                lpips_render = compute_img_metric(images_gs_m, imgs_render_mid, 'lpips')
                with open(test_metric_file, 'a') as outfile:
                    outfile.write(f"m: iter{i}: MSE:{mse_render.item():.8f} PSNR:{psnr_render.item():.8f}"
                                f" SSIM:{ssim_render.item():.8f} LPIPS:{lpips_render.item():.8f}\n")
            
            if images_gs_f != None:
                mse_render = compute_img_metric(images_gs_f, imgs_render_start, 'mse')
                psnr_render = compute_img_metric(images_gs_f, imgs_render_start, 'psnr')
                ssim_render = compute_img_metric(images_gs_f, imgs_render_start, 'ssim')
                lpips_render = compute_img_metric(images_gs_f, imgs_render_start, 'lpips')
                with open(test_metric_file, 'a') as outfile:
                    outfile.write(f"f:  iter{i}: MSE:{mse_render.item():.8f} PSNR:{psnr_render.item():.8f}"
                                f" SSIM:{ssim_render.item():.8f} LPIPS:{lpips_render.item():.8f}\n")

        # render high-framerate video of first frame
        if i % args.i_video == 0 and i > 0:
            args.train_state == 'val'
            with torch.no_grad():
                rgbs, disps = render_video_test(i, graph, test_render_poses, H//args.render_downsample, W//args.render_downsample, K/args.render_downsample, args)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_test_render_poses_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        # render video of novel view synthesis
        if i % args.i_video == 0 and i > 0:
            args.train_state == 'val'
            bds = np.array([1 / 0.75, 150 / 0.75])
            optimized_se3 = graph.se3.params.weight.data
            optimized_pose = se3_to_SE3_N(optimized_se3)
            optimized_pose = torch.cat([optimized_pose, torch.tensor([H, W, focal]).reshape([1,3,1]).repeat(optimized_pose.shape[0], 1, 1)], -1)
            optimized_pose = optimized_pose.cpu().numpy()
            render_poses = regenerate_pose(optimized_pose, bds, recenter=True, bd_factor=.75, spherify=False, path_zflat=False)
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_video_test(i, graph, render_poses, H//args.render_downsample, W//args.render_downsample, K/args.render_downsample, args)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        args.train_state == 'train'

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = config_parser()
    args = parser.parse_args()
    train(args = args)
