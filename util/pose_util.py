#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import math
import copy
from einops import rearrange

import math
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar

def _quat_to_xy_heading(quat):
    direction_vector = np.array([0, 0, -1])
    heading_vector = quaternion_rotate_vector(quat.inverse(), direction_vector)
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi

def parse_pose(pose):
    apos, arot = pose[:3], pose[3:]
    arot_q = quaternion_from_coeff(arot)
    heading = _quat_to_xy_heading(arot_q)
    return [apos[0], apos[2], heading]

class CameraProjection:

    def __init__(self, img_sz):
        self.cam_params = (img_sz//2, img_sz//2, (img_sz//2)-1, (img_sz//2)-1) # fx, fy, cx, cy
        self.rot_cache = {}
        
    def get_rotation_matrix(self, axis, angle_deg):

        if (axis, angle_deg) not in self.rot_cache:

            angle_rad  = 2 * math.pi * (angle_deg / 360.0)

            c = math.cos(angle_rad)
            s = math.sin(angle_rad)

            if axis=='x':
                R = torch.Tensor(
                    [[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

            elif axis=='y':
                R = torch.Tensor(
                        [[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])


            elif axis=='z':
                R = torch.Tensor(
                    [[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

            self.rot_cache[(axis, angle_deg)] = R


        return self.rot_cache[(axis, angle_deg)]

    def image_to_world(self, agent_pose, depth_inputs):

        agent_pose = torch.Tensor(agent_pose)
        depth_inputs = torch.from_numpy(depth_inputs).unsqueeze(0) # depth in meters (1, 300, 300)

        agent_pose = agent_pose.unsqueeze(0)
        depth_inputs = depth_inputs.unsqueeze(0)

        fx, fy, cx, cy = self.cam_params
        bs, _, imh, imw = depth_inputs.shape
        device          = depth_inputs.device

        # 2D image coordinates
        x               = rearrange(torch.arange(0, imw), 'w -> () () () w')
        y               = rearrange(torch.arange(0, imh), 'h -> () () h ()')
        x, y            = x.float().to(device), y.float().to(device)

        xx              = (x - cx) / fx
        yy              = (y - cy) / fy

        # 3D real-world coordinates (in meters)
        Z               = -depth_inputs
        X               = xx * depth_inputs # (B, 1, imh, imw)
        Y               = yy * depth_inputs # (B, 1, imh, imw)

        P = torch.cat([X, Y, Z], 1) # (B, 3, imh, imw)

        P = rearrange(P, 'b p h w -> b p (h w)') # (B, 3, h*w) # matrix mult time

        R = torch.stack([self.get_rotation_matrix('y', rot) for rot in agent_pose[:, 3]]) # (B, 3, 3)
        
        P0 = agent_pose[:, 0:3] # (B, 3)
        P0[:, 1] = -P0[:, 1] # negative y
        P0 = P0.unsqueeze(-1) # (B, 3, 1)

        R = R.to(depth_inputs.device)
        P = torch.bmm(R, P) + P0 # (B, 3, 3) * (B, 3, h*w) + (B, 3, 1) --> (B, 3, h*w)
        P = rearrange(P, 'b p (h w) -> b p h w', h=imh, w=imw)

        return P

