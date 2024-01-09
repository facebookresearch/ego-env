#  Copyright (c) Meta Platforms, Inc. and affiliates.
#  All rights reserved.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

BUCKET=https://dl.fbaipublicfiles.com/ego-env
function dls3 {
	mkdir -p `dirname $1`
	wget -O $1 $BUCKET/$1
}

function dls3_unzip {
	mkdir -p `dirname $1`
	wget -O $1 $BUCKET/$1
	unzip $1 -d `dirname $1`
	rm $1
}

# download walkthrough data: full episode list + sample metadata for 50 episodes
dls3 data/walkthrough_data/hm3d/v1/episode_list.pth
dls3_unzip data/walkthrough_data/hm3d/v1_sample.zip
echo 'Downloaded walkthrough sample data'

# download pre-trained checkpoints
dls3_unzip checkpoints.zip
dls3 walkthrough_generation/models/mmdet_qinst_hm3d.pth
echo 'Downloaded model checkpoints'

# download task annotations and metadata
dls3_unzip data/annotations.zip
dls3 data/housetours/traj_metadata.pth
echo 'Downloaded task annotations'


