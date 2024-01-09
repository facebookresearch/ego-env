#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import numpy as np

def _format_pose(pose):
    """
    Args:
        pose: (..., 4) Tensor containing x, y, heading, time
    """
    x, y, heading, time = torch.unbind(pose, dim=-1)
    cos_heading, sin_heading = torch.cos(heading), torch.sin(heading)
    e_time = torch.exp(-time)
    formatted_pose = torch.stack([x, y, cos_heading, sin_heading, e_time], -1)
    return formatted_pose

def _encode_pose(agent_pose, memory_pose):
    """
    Args:
        agent_pose: (bs, 4) Tensor containing x, y, heading, time
        memory_pose: (M, bs, 4) Tensor containing x, y, heading, time
    """
    agent_xyh, agent_t = agent_pose[..., :3], agent_pose[..., 3:4]
    memory_xyh, memory_t = memory_pose[..., :3], memory_pose[..., 3:4]

    # Compute relative poses
    agent_rel_xyh = _compute_relative_pose(agent_xyh, agent_xyh)
    agent_rel_pose = torch.cat([agent_rel_xyh, agent_t], -1)
    memory_rel_xyh = _compute_relative_pose(agent_xyh.unsqueeze(0), memory_xyh)
    memory_rel_pose = torch.cat([memory_rel_xyh, memory_t], -1)

    # Format pose
    agent_pose_formatted = _format_pose(agent_rel_pose)
    memory_pose_formatted = _format_pose(memory_rel_pose)

    return agent_pose_formatted, memory_pose_formatted

def _compute_relative_pose(pose_a, pose_b):
    """
    Computes the pose_b - pose_a in pose_a's coordinates.
    Args:
        pose_a: (..., 3) Tensor of x in meters, y in meters, heading in radians
        pose_b: (..., 3) Tensor of x in meters, y in meters, heading in radians
    Expected pose format:
        At the origin, x is forward, y is rightward,
        and heading is measured from x to -y.
    """
    # Negate the heading to get angle from x to y
    heading_a = -pose_a[..., 2]
    heading_b = -pose_b[..., 2]
    # Compute relative pose
    r_ab = torch.norm(pose_a[..., :2] - pose_b[..., :2], dim=-1)
    phi_ab = torch.atan2(pose_b[..., 1] - pose_a[..., 1], pose_b[..., 0] - pose_a[..., 0])
    phi_ab = phi_ab - heading_a
    x_ab = r_ab * torch.cos(phi_ab)
    y_ab = r_ab * torch.sin(phi_ab)
    heading_ab = heading_b - heading_a
    # Normalize angles to lie between -pi to pi
    heading_ab = torch.atan2(torch.sin(heading_ab), torch.cos(heading_ab))
    # Negate the heading to get angle from x to -y
    heading_ab = -heading_ab

    return torch.stack([x_ab, y_ab, heading_ab], -1) # (..., 3)


# https://stackoverflow.com/questions/65565461/how-to-map-element-in-pytorch-tensor-to-id
def bucketize(a: torch.Tensor, ids: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mapping = {k.item(): v.item() for k, v in zip(a, ids)}

    # From `https://stackoverflow.com/questions/13572448`.
    palette, key = zip(*mapping.items())
    key = torch.tensor(key)
    palette = torch.tensor(palette)

    index = torch.bucketize(b.ravel(), palette)
    remapped = key[index].reshape(b.shape)

    return remapped

def strided_window_frames(query_inds, total_frames, mem_size, win_size, train_mode):

    if train_mode:
        win_size = np.random.choice([64, 128, 256, 512])

    # find win_size windows centered around query_inds
    centers = query_inds
    if train_mode:
        offset = torch.randint(-win_size//2, win_size//2+1, (query_inds.shape[0],))
        centers = (query_inds + offset).clip(0, total_frames)

    win_start_frames = (centers - win_size//2).clip(0, total_frames - win_size) # (T,)

    # sample from each window
    inter_offset = torch.zeros(query_inds.shape[0], mem_size)
    if train_mode:
        stride = win_size // mem_size
        if stride > 1:
            inter_offset = torch.randint(-stride//2, stride//2+1, (query_inds.shape[0], mem_size))

    mem_blocks = torch.stack([torch.linspace(start, start + win_size - 1, mem_size) for start in win_start_frames], 0) 
    mem_blocks = (inter_offset + mem_blocks).clip(0, total_frames-1).long() # (T, S)

    # list of frames to load \in [0, total_frames] (Sp,)
    frame_inds = mem_blocks.unique()

    # float inds wrt each individual mem_block query
    fmin, fmax = mem_blocks.amin(1, keepdim=True), mem_blocks.amax(1, keepdim=True)
    rel_frame_pos = (mem_blocks.float() - fmin) / (fmax - fmin)
    rel_query_pos = (query_inds.float() - fmin.squeeze(1)) / (fmax - fmin).squeeze(1)

    # re-map mem_blocks to frame_inds
    # among those frames \in [0, Sp], what inds to load per query_idx (T, S)
    mem_blocks = bucketize(frame_inds, torch.arange(len(frame_inds)), mem_blocks)

    # sometimes the query overflows because of the inter_offset
    rel_query_pos = rel_query_pos.clip(0, 1)

    # re-normalize for max_len of 512
    max_len = 512
    if win_size <= 512:
        rel_frame_pos = rel_frame_pos * win_size / max_len
        rel_query_pos = rel_query_pos * win_size / max_len

    return frame_inds, mem_blocks, rel_frame_pos, rel_query_pos