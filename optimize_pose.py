import torch.nn

import cubicSpline
import nerf


class SE3(torch.nn.Module):
    def __init__(self, img_num):
        super().__init__()
        self.params = torch.nn.Embedding(img_num, 6)

class Model(nerf.Model):
    def __init__(self, se3):
        super().__init__()
        self.se3_init = se3

    def build_network(self, args):
        self.graph = Graph(args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True)
        low, high = 1e-4, 1e-3
        if args.SplineModel == "Cubic":
            pose_params = torch.cat([self.se3_init[:1], self.se3_init, self.se3_init[-1:], self.se3_init[-1:]], dim=0)
        # use 4 control knots to get interpolatted poses between middle 2 knots, eg: control knots 0,1,2,3, get splined trajectory between knots 1 and 2, control knots 1,2,3,4, get splined trajectory between knots 2 and 3, 
        # thus divide long trajectory into N small segmented trajectories, we need N+3 control knots
        elif args.SplineModel == "Linear":
            pose_params = torch.cat([self.se3_init, self.se3_init[:1,:6]], dim=0)
            
        pose_params = pose_params + (high - low) * torch.randn_like(pose_params) + low
        self.graph.se3 = SE3(pose_params.shape[0])
        self.graph.se3.params.weight.data = torch.nn.Parameter(pose_params)

        return self.graph

    def setup_optimizer(self, args):
        grad_vars = list(self.graph.nerf.parameters())
        if args.N_importance > 0:
            grad_vars += list(self.graph.nerf_fine.parameters())
        self.optim = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

        grad_vars_se3 = list(self.graph.se3.parameters())
        self.optim_se3 = torch.optim.Adam(params=grad_vars_se3, lr=args.lrate)

        return self.optim, self.optim_se3


class Graph(nerf.Graph):
    def __init__(self, args, D=8, W=256, input_ch=63, input_ch_views=27, output_ch=4, skips=[4], use_viewdirs=True):
        super().__init__(args, D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)
        self.pose_eye = torch.eye(3, 4)

    def get_pose(self, i, args, indices, H):

        img_idx = indices[:,0]
        pixel_y = indices[:,1]

        ts = pixel_y * args.readout_time

        if args.SplineModel == "Cubic":
            pose0 = self.se3.params.weight[img_idx, :6]
            pose1 = self.se3.params.weight[img_idx + 1, :6]
            pose2 = self.se3.params.weight[img_idx + 2, :6]
            pose3 = self.se3.params.weight[img_idx + 3, :6]

            spline_poses = cubicSpline.SplineN_cubic(pose0, pose1, pose2, pose3, ts, args.period)
        
        elif args.SplineModel == "Linear":

            se3_start = self.se3.params.weight[:-1, :6][img_idx]
            se3_end = self.se3.params.weight[1:, :6][img_idx]

            spline_poses = cubicSpline.SplineN_linear(se3_start, se3_end, ts, args.period)

        return spline_poses

    def get_gt_pose(self, poses, args):
        a = self.pose_eye
        return poses
