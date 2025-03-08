import os
import glob
import numpy as np
from voxelmorph import py

def volgen(
    vol_names,
    batch_size=1,
    segs=None,
    np_var='vol',
    pad_shape=None,
    resize_factor=1,
    add_feat_axis=True
):
    # convert glob path to filenames
    if isinstance(vol_names, str):
        if os.path.isdir(vol_names):
            vol_names = os.path.join(vol_names, '*')
        vol_names = glob.glob(vol_names)

    cases = {}
    for case_path in vol_names:
        case_path_name = os.path.basename(case_path)
        case_name = case_path_name.split('_')[0]
        if case_name not in cases:
            cases[case_name] = []
        cases[case_name].append(case_path)

    print('detect case number:', len(cases))

    cases_names = list(cases.keys())

    while True:
        # choose one case
        case_name = np.random.choice(cases_names)

        load_params = dict(np_var=np_var, add_batch_axis=True, add_feat_axis=add_feat_axis,
                           pad_shape=pad_shape, resize_factor=resize_factor)

        indices = np.random.randint(len(cases[case_name]), size=batch_size)        
        imgs = [py.utils.load_volfile(cases[case_name][i], **load_params) for i in indices]
        inputs = np.concatenate(imgs, axis=0)

        indices = np.random.randint(len(cases[case_name]), size=batch_size)        
        imgs = [py.utils.load_volfile(cases[case_name][i], **load_params) for i in indices]
        labels = np.concatenate(imgs, axis=0)

        yield inputs, labels

def scan_to_scan(vol_names, bidir=False, batch_size=1, prob_same=0, no_warp=False, **kwargs):
    zeros = None
    gen = volgen(vol_names, batch_size=batch_size, **kwargs)
    while True:
        scan1, scan2 = next(gen)

        assert scan1.shape[3] >= 64, 'make sure the #slice of input >= 64!'
        assert scan2.shape[3] >= 64, 'make sure the #slice of input >= 64!'

        scan1 = scan1[:, :, :, :64, :]
        scan2 = scan2[:, :, :, :64, :]
        # (1, 512, 512, 64, 1)

        # some induced chance of making source and target equal
        if prob_same > 0 and np.random.rand() < prob_same:
            if np.random.rand() > 0.5:
                scan1 = scan2
            else:
                scan2 = scan1

        # cache zeros
        if not no_warp and zeros is None:
            shape = scan1.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

        invols = [scan1, scan2]
        outvols = [scan2, scan1] if bidir else [scan2]
        if not no_warp:
            outvols.append(zeros)

        yield (invols, outvols)