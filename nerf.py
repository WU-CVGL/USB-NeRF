import torch
import torch.nn as nn
import torch.nn.functional as F

from run_nerf import *
import cubicSpline

PIXELS_EVERY_POSE = 7
max_iter = 200000

def barf(barf_i, ps_embedded, L, args):  # ps_embedded: [, 6L_new]
    L_new = L // 6
    progress = (barf_i-1)/max_iter
    start, end = args.barf_start, args.barf_end    # 0.1 0.5
    alpha = (progress - start) / (end - start) * L_new
    k = torch.arange(L_new)
    weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
    shape = ps_embedded.shape
    ps_embedded = (ps_embedded.view(-1,L_new)*weight).view(*shape)    # 从 args.barf_start * max_iter 到 args.barf_end * max_iter 逐渐引入高频分量
    return ps_embedded


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(args, multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': False if args.barf else True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Model():
    def __init__(self):
        super().__init__()

    def build_network(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance>0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        return self.optim
        # here: 这里还没有更新，每次都要更新学习率


class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=False):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # network
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
                                        range(D - 1)])

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    # positional encoding和nerf的mlp
    def forward(self, barf_i, pts, viewdirs, args):
        # step1: positional encoding
        # create positional encoding
        embed_fn, input_ch = get_embedder(args, args.multires, args.i_embed)  # xyz 公式4
        input_ch_views = 0
        embeddirs_fn = None
        if args.use_viewdirs:
            embeddirs_fn, input_ch_views = get_embedder(args, args.multires_views, args.i_embed)
        # forward positional encoding
        pts_flat = torch.reshape(pts, [-1, pts.shape[-1]])  # [N_rands x 64, 3] (pose_num * rays_num)
        embedded = embed_fn(pts_flat)  # [N_rands x 64, 63] if barf: [..., 60]

        if args.barf:
            embedded = barf(barf_i, embedded, input_ch, args)
            embedded = torch.cat([pts_flat, embedded], -1)  # [..., 63]

        if viewdirs is not None:
            # embedded_dirs:[1024x64, 27]
            input_dirs = viewdirs[:, None].expand(pts.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = embeddirs_fn(input_dirs_flat)  # [N_rands x 64, 27] if barf: [..., 24]
            if args.barf:
                embedded_dirs = barf(barf_i, embedded_dirs, input_ch_views, args)
                embedded_dirs = torch.cat([input_dirs_flat, embedded_dirs], -1)  # [..., 27]
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # step2
        input_pts, input_views = torch.split(embedded, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)  # [N, 4(RGBA A is sigma)]
        else:
            outputs = self.output_linear(h)

        # [1024, 64, 4]
        outputs = torch.reshape(outputs, list(pts.shape[:-1]) + [outputs.shape[-1]])

        return outputs  # [N, 4(RGBA A is sigma)]

    def raw2output(self, raw, z_vals, rays_d, raw_noise_std=1.0):

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:,
                          :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        # fixme
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        # disp_map = torch.max(1e-6 * torch.ones_like(depth_map), depth_map / (torch.sum(weights, -1) + 1e-6))
        acc_map = torch.sum(weights, -1)

        sigma = F.relu(raw[..., 3] + noise)

        return rgb_map, disp_map, acc_map, weights, depth_map, sigma


class Graph(nn.Module):

    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=False):
        super().__init__()
        self.nerf = NeRF(D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)
        if args.N_importance > 0:
            self.nerf_fine = NeRF(D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)
        

    def eval_novel_view(self, img_idx, H, W, K, args):
        """
            only render one val view.
            img_idx: index of novel view
            H: image height
            W: image width
            K: camera intrinscis
        
        """
        indices = torch.floor(
            torch.rand((args.total_pixel, 3), device=device)
            * torch.tensor([img_idx, H, W], device=device)
        ).long()

        indices[:,0] = img_idx
        spline_poses = self.get_pose_idx(img_idx).unsqueeze(0).repeat(args.total_pixel, 1, 1)

        ray_idx = indices[:,1] * W + indices[:,2]
        ret = self.render(0, spline_poses, ray_idx, H, W, K, args, ray_idx_tv=None, training=True)

        return ret, indices


    def test_novel_view(self, images_num, H, W, K, args):
        """
            test all novel views.
        """
        indices = torch.floor(
                torch.rand((args.total_pixel, 3), device=device)
                * torch.tensor([images_num, H, W], device=device)
        ).long()

        spline_poses = self.get_pose_idx(indices[:,0])

        ray_idx = indices[:,1] * W + indices[:,2]
        ret = self.render(0, spline_poses, ray_idx, H, W, K, args, ray_idx_tv=None, training=True)

        test_poses = self.get_pose_idx(torch.arange(images_num))

        return ret, indices, test_poses
    

    def forward(self, i, images_num, H, W, K, args):
        indices = torch.floor(
                torch.rand((args.total_pixel, 3), device=device)
                * torch.tensor([images_num, H, W], device=device)
            ).long()

        spline_poses = self.get_pose(i, args, indices, H)

        ray_idx = indices[:,1] * W + indices[:,2]
        ret = self.render(i, spline_poses, ray_idx, H, W, K, args, ray_idx_tv=None, training=True)


        # if args.render_rolling_shutter:
        #     spline_num = H // row
        #     test_poses = self.get_pose(i, torch.arange(self.se3.start.weight.shape[0]), args,
        #                                torch.arange(spline_num).unsqueeze(0).repeat(self.se3.start.weight.shape[0],
        #                                                                             1), H / row,
        #                                args.trajectory_seg_num)  # [0,1,...,34] 代表所有图片   [0,1,...,H//10-1]
        #     return test_poses

        if i % args.i_video == 0 and i > 0:
            img_idx = torch.arange(images_num).unsqueeze(1).repeat(1, H).flatten().unsqueeze(1)
            pixels_y = torch.arange(H).unsqueeze(0).repeat(images_num,1).flatten().unsqueeze(1)
            test_indices = torch.concat([img_idx, pixels_y], dim=1)
            test_poses = self.get_pose(i, args, test_indices, H)
            i_render_mid = torch.arange(images_num) * H + H // 2
            i_render_start = torch.arange(images_num) * H
            i_render_end = torch.arange(images_num) * H + H - 1

            test_render_poses = test_poses[torch.arange(59) * 8]
            return ret, indices, test_poses[i_render_mid], test_poses[i_render_start], test_poses[i_render_end], test_render_poses

        elif i % args.i_img == 0 and i > 0:
            img_idx = torch.arange(images_num).unsqueeze(1).repeat(1, H).flatten().unsqueeze(1)
            pixels_y = torch.arange(H).unsqueeze(0).repeat(images_num,1).flatten().unsqueeze(1)
            test_indices = torch.concat([img_idx, pixels_y], dim=1)
            test_poses = self.get_pose(i, args, test_indices, H)
            i_render_mid = torch.arange(images_num) * H + H//2
            i_render_start = torch.arange(images_num) * H
            i_render_end = torch.arange(images_num) * H + H - 1

            return ret, indices, test_poses[i_render_mid], test_poses[i_render_start], test_poses[i_render_end]

        else:
            return ret, indices


    def get_pose(self, i, img_idx, args, pose_nums, H, trajectory_seg_num):    # 父类 get_pose 会被子类重写，所以后续调用的 get_pose 都为 cubic_spline 之后的 pose

        return i

    def get_gt_pose(self, poses, args):

        return poses

    def render(self, barf_i, poses, ray_idx, H, W, K, args, near=0., far=1., ray_idx_tv=None, training=False):

        if training:
            j = ray_idx // W    # Y
            i = ray_idx % W    # X
            if isinstance(args.xys_remap, np.ndarray) and args.train_state == 'train':
                x_rect = torch.from_numpy(args.xys_remap[:,:,0]).to(device=ray_idx.device)[j,i]
                y_rect = torch.from_numpy(args.xys_remap[:,:,1]).to(device=ray_idx.device)[j,i]
            else:
                x_rect = i
                y_rect = j

            # poses = poses.unsqueeze(1).repeat(1, args.pixels_every_pose, 1, 1).reshape(-1, 3, 4) # repeat the poses for the times of pixels per pose   [poses_num, 3, 4] --> [poses_num * args.pixels_every_pose, 3, 4]
            rays_o_, rays_d_ = get_specific_rays(x_rect, y_rect, K, poses)
            rays_o_d = torch.stack([rays_o_, rays_d_], 0)  # [ro+rd, rays_num, 3]
            batch_rays = torch.permute(rays_o_d, [1, 0, 2])  # # [rays_num, ro+rd, 3]

        else:
            rays_list = []

            for p in poses[:, :3, :4]:
                rays_o_, rays_d_ = get_rays(H, W, K, p)
                rays_o_d = torch.stack([rays_o_, rays_d_], 0)
                rays_list.append(rays_o_d)

            rays = torch.stack(rays_list, 0)          # [N(poses), ro+rd, H, W, 3]
            rays = rays.reshape(-1, 2, H * W, 3)      # [N(poses), ro+rd, H * W, 3]    把原始的 矩阵图 展开成一行
            rays = torch.permute(rays, [0, 2, 1, 3])  # [N(poses), H * W, ro+rd, 3]

            batch_rays = rays[:, ray_idx]  # [N, N_rand//N, ro+rd, 3]


        batch_rays = batch_rays.reshape(-1, 2, 3)  # here: 交换位置？[N*N_rand//N, ro+rd, 3]
        batch_rays = torch.transpose(batch_rays, 0, 1)  # [ro+rd, N*N_rand//N, 3]

        if ray_idx_tv is not None:
            batch_rays_tv = rays[0, ray_idx_tv]
            batch_rays_tv = batch_rays_tv.reshape(-1, 2, 3)
            batch_rays_tv = torch.transpose(batch_rays_tv, 0, 1)
            batch_rays = torch.cat([batch_rays, batch_rays_tv], 1)

        # get standard rays
        rays_o, rays_d = batch_rays
        if args.use_viewdirs:
            viewdirs = rays_d
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape
        if args.ndc:
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        # rays_o = torch.reshape(rays_o, [-1, 3]).float()
        # rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)

        if args.use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)  # this is final rays

        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # [N_rays, 3] each
        viewdirs = rays[:, -3:] if rays.shape[-1] > 8 else None
        bounds = torch.reshape(rays[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]

        t_vals = torch.linspace(0., 1., steps=args.N_samples)
        z_vals = near * (1. - t_vals) + far * (t_vals)
        z_vals = z_vals.expand([N_rays, args.N_samples])

        # perturb
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        # step4: forward_rays: nerf.forward: 输入是pts和viewdirs 输出:raw here:n y
        raw_output = self.nerf.forward(barf_i, pts, viewdirs, args)

        # step5: raw2output: nerf.raw2output 输入:(raw, z_vals, rays_d, raw_noise_std=1.0) y
        rgb_map, disp_map, acc_map, weights, depth_map, sigma = self.nerf.raw2output(raw_output, z_vals, rays_d)

        # step6: if fine, 重复step3-5
        if args.N_importance > 0:
            rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
            # step3
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], args.N_importance)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]
            # step4 & step5
            raw_output = self.nerf_fine.forward(barf_i, pts, viewdirs, args)
            rgb_map, disp_map, acc_map, weights, depth_map, sigma = self.nerf_fine.raw2output(raw_output, z_vals,
                                                                                              rays_d)

        ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
        if args.N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['sigma'] = sigma

        return ret

    @torch.no_grad()
    def render_video(self, barf_i, poses, H, W, K, args):
        all_ret = {}
        ray_idx = torch.arange(0, H*W)
        for i in range(0, ray_idx.shape[0], args.chunk):
            ret = self.render(barf_i, poses, ray_idx[i:i+args.chunk], H, W, K, args)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list([H, W]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)
        return all_ret