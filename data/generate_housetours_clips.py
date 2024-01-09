import os
#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

import torch
import tqdm
from moviepy.editor import VideoFileClip

def generate_clip(entry):
    vclip = VideoFileClip(f'data/housetours/videos/{entry["video_uid"]}.mp4')
    vclip = vclip.subclip(entry['start_time'], entry['end_time'])
    vclip.write_videofile(
        f'data/housetours/clips/{entry["clip_uid"]}.mp4',
        audio=False,
        threads=8,
        logger=None
    )

os.makedirs('data/housetours/clips/', exist_ok=True)
traj_data = torch.load(f'data/housetours/traj_metadata.pth')
for entry in tqdm.tqdm(traj_data):
    generate_clip(entry)
    break