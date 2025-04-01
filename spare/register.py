import os
import argparse
import math

import numpy as np
import nibabel as nib
import torch

os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm   # nopep8

parser = argparse.ArgumentParser()
parser.add_argument('--moving', required=True, help='moving image (source) filename')
parser.add_argument('--fixed', required=True, help='fixed image (target) filename')
parser.add_argument('--moved', default=None, help='warped image output filename')
parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
parser.add_argument('--warp', default=None, help='output warp deformation filename')
parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
args = parser.parse_args()

if args.gpu and (args.gpu != '-1'):
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    device = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# load moving and fixed images
add_feat_axis = not args.multichannel
moving = vxm.py.utils.load_volfile(args.moving, add_batch_axis=True, add_feat_axis=add_feat_axis)
fixed, fixed_affine = vxm.py.utils.load_volfile(
    args.fixed, add_batch_axis=True, add_feat_axis=add_feat_axis, ret_affine=True)

# load and set up model
model = vxm.networks.VxmDense(
    inshape=(512, 512, 64),
    nb_unet_features=[[16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]],
    bidir=False,
    int_steps=7,
    int_downsize=2
)

assert os.path.exists(args.model)
model_state_dict = torch.load(args.model, map_location=device)
model.load_state_dict(model_state_dict)
print('load checkpoint from ', args.model)

model.to(device)
model.eval()

# set up tensors and permute
input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)

slices_num = input_moving.shape[-1]
assert slices_num >= 64

moveds = []
warps = []

for i in range(math.ceil(slices_num / 64)):
    min_idx = 64 * i
    max_idx = min_idx + 64
    exceed = 0
    if max_idx > slices_num:
        exceed = max_idx - slices_num
        max_idx -= exceed
        min_idx -= exceed


    # choose current 64
    input_moving_i = input_moving[..., min_idx: max_idx]
    input_fixed_i = input_fixed[..., min_idx: max_idx]

    # predict
    with torch.no_grad():
        moved, warp = model(input_moving_i, input_fixed_i, registration=True)

    if exceed > 0:
        moved = moved[..., exceed: ]
        warp = warp[..., exceed: ]

    moveds.append(moved.detach().cpu())
    warps.append(warp.detach().cpu())

    del moved
    del warp

moved = torch.cat(moveds, dim=4)
warp = torch.cat(warps, dim=4)

# save moved image
file_name = os.path.basename(args.moving)
case_name = file_name.split('_')[0]
if args.moved is None:
    moved_name = f'{case_name}_moved.nii.gz'
    args.moved = os.path.join('experiment', moved_name)

# save warp
if args.warp is None:
    warp_name = f'{case_name}_warp.nii.gz'
    args.warp = os.path.join('experiment', warp_name)

moved = moved.detach().cpu().numpy().squeeze()
print('moved shape:', moved.shape)
vxm.py.utils.save_volfile(moved, args.moved, fixed_affine)

warp = warp.detach().cpu().numpy().squeeze()
print('warp shape:', warp.shape)
vxm.py.utils.save_volfile(warp, args.warp, fixed_affine)