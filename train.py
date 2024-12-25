import os
import random
import argparse
import time
import numpy as np
import torch

from loguru import logger

logger.remove(handler_id=None)
logger.add(
    sink='log/voxelmorph.log',
    level='INFO',
    encoding='utf-8',
    format="{time:YYYY-MM-DD HH:mm:ss} | {message} "
)

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8

import dataloader

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', required=True, help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separatedatlas (default: 0)')
parser.add_argument('--atlas', default=None, help='GPU ID number(s), comma-separatedatlas (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir

from tqdm import tqdm, trange

if __name__ == '__main__':
    # load and prepare training data
    train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                            suffix=args.img_suffix)
    assert len(train_files) > 0, 'Could not find any training data.'

    # no need to append an extra feature axis if data is multichannel
    add_feat_axis = not args.multichannel

    # realize your own loader
    generator = dataloader.scan_to_scan(
        train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)
    # if args.atlas:
    #     # scan-to-atlas generator
    #     atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
    #                                     add_batch_axis=True, add_feat_axis=add_feat_axis)
    #     generator = vxm.generators.scan_to_atlas(train_files, atlas,
    #                                             batch_size=args.batch_size, bidir=args.bidir,
    #                                             add_feat_axis=add_feat_axis)
    # else:
    #     # scan-to-scan generator
    #     generator = vxm.generators.scan_to_scan(
    #         train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)


    # extract shape from sampled input
    inshape = next(generator)[0][0].shape[1:-1]

    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    # unet architecture
    enc_nf = args.enc if args.enc else [16, 32, 32, 32]
    dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]


    # otherwise configure new model
    print('init model, inshape:', inshape)
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    ) 

    if args.load_model:
        # load initial model (if specified)
        assert os.path.exists(args.load_model)
        model_state_dict = torch.load(args.load_model, map_location=device)
        model.load_state_dict(model_state_dict)
        print('load checkpoint from ', args.load_model)

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # prepare image loss
    if args.image_loss == 'ncc':
        image_loss_func = vxm.losses.NCC().loss
    elif args.image_loss == 'mse':
        image_loss_func = vxm.losses.MSE().loss
    else:
        raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

    # need two image loss functions if bidirectional
    if bidir:
        losses = [image_loss_func, image_loss_func]
        weights = [0.5, 0.5]
    else:
        losses = [image_loss_func]
        weights = [1]

    # prepare deformation loss
    losses += [vxm.losses.Grad('l2', loss_mult=args.int_downsize).loss]
    weights += [args.weight]

    if args.load_model:
        ckp_name = os.path.basename(args.load_model)
        ckp_id = int(ckp_name.strip('.pt'))
        args.initial_epoch = ckp_id + 1

    # training loops
    for epoch in range(args.initial_epoch, args.epochs + args.initial_epoch):

        # save model checkpoint
        if epoch % 5 == 0:
            save_path = os.path.join(model_dir, '%04d.pt' % epoch)
            torch.save(model.state_dict(), save_path)

        epoch_loss = []
        epoch_total_loss = []

        tbar = trange(args.steps_per_epoch, colour='yellow', ncols=80)
        tbar.set_description_str(f'epoch: {epoch}, loss=unknown')
        for step in tbar:
            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(generator)
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]

            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs)

            # calculate total loss
            loss = 0
            loss_list = []
            for n, loss_function in enumerate(losses):
                curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tbar.set_description_str(f'epoch: {epoch}, loss={round(loss.item(), 4)}')

        # print epoch info
        epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
        losses_info = ', '.join(['%.4e' % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)
        logger.info(loss_info)

    # final model save
    save_path = os.path.join(model_dir, '%04d.pt' % args.epochs)
    torch.save(model.state_dict(), save_path)