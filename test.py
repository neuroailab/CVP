# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import torch
from cfgs.test_cfgs import TestOptions
from utils import model_utils
import cvp.vis as vis_utils
from cvp.logger import Logger
from cvp.losses import LossManager

from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def main(args, model, save_path):
    torch.manual_seed(123)
    dt = 10

    data_loader = model_utils.build_loaders(args)  # change to image
    # loss_mng = LossManager(args)
    print('data loaded')

    cnt = 0
    sum_losses = {'bbox_loss':0., 'appr_pixel_loss':0.}

    img_feats = []
    feats = []
    print('batch size', args.batch_size)
    for batch in tqdm(data_loader):
        with torch.no_grad():
            #predictions = model.forward_inception(batch, args.dt)
            predictions = model.forward_with_reality(batch, 10)

        img_feats.append(torch.stack(predictions['src_feats']).squeeze().cpu().numpy())
        feats.append(predictions['appr'].squeeze().cpu().numpy().reshape(6, -1, 32*2*2))

        bbox_loss = F.mse_loss(predictions['bbox'], batch['bbox'][1:, ...])
        sum_losses['bbox_loss'] += bbox_loss.item()
        cnt += args.batch_size

    print(sum_losses, cnt)
    for k, s in sum_losses.items():
        print(k, s/cnt)

    np.save(save_path, np.array(feats))
    np.save(save_path[:-4]+'_img.npy', np.array(img_feats))
    print('saved at {}'.format(save_path))

class BinaryClassification(torch.nn.Module):
    def __init__(self, input_dimension):
            super().__init__()
            self.linear = torch.nn.Linear(input_dimension, 1)
    def forward(self, input_dimension):
            return self.linear(input_dimension)

if __name__ == '__main__':
    args = TestOptions().parse()
    args.l1_dst_loss_weight = 1.
    args.bbox_loss_weight = 1
    args.l1_src_loss_weight = 1.

    datasets = args.dataset.split(',')
    args.dataset = datasets[0]

    model = model_utils.build_all_model(args)  # CNN, GCN, Encoder, Decoder

    print('model loaded')
    save_path = '{}.npy'.format(args.checkpoint[:-4])
    if not os.path.exists(save_path):
        main(args, model, save_path)

    args.dataset = datasets[1]
    epochs = args.checkpoint.split('r')[-1].split('.')[0]
    human_save_path = '{}/feats_{}.npy'.format(args.dataset, epochs)
    if not os.path.exists(human_save_path):
        main(args, model, human_save_path)

    import json
    import numpy as np
    from tqdm import tqdm
    from sklearn.linear_model import LogisticRegression

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_path = '{}.npy'.format(args.checkpoint[:-4])

    with open('{}/labels.json'.format(datasets[0]), 'r') as f:
        y_train = np.array(json.load(f)).astype(float)
    with open('{}/labels.json'.format(datasets[1]), 'r') as f:
        y_test = np.array(json.load(f)).astype(float)

    
    X_train = np.load(save_path)
    X_train = X_train.reshape(X_train.shape[0], 6, -1)
    y_train = y_train[:X_train.shape[0], ...]
    # X_test = np.load('results/test_10000.npy')
    X_test = np.load(human_save_path)
    X_test = X_test.reshape(X_test.shape[0], 6, -1)

    target_field = 1
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    y_train = y_train[:, target_field]
    y_test = y_test[:, target_field]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    print('fitting..')
    clf = LogisticRegression(random_state=0,class_weight='balanced').fit(X_train, y_train)
    print('=====')
    print('Table2, Test Acc {:.4f}'.format(clf.score(X_test, y_test)))
    print('=====')

    save_path = save_path[:-4]+'_img.npy'
    human_save_path = human_save_path[:-4]+'_img.npy'
    X_train = np.load(save_path)
    X_test = np.load(human_save_path)

    X_train = X_train.reshape(X_train.shape[0], 10, -1)
    X_test = X_test.reshape(X_test.shape[0], 10, -1)
    X_train = X_train[:, :4, :].reshape(X_train.shape[0], -1)
    X_test = X_test[:, :4, :].reshape(X_test.shape[0], -1)

    with open('{}/labels.json'.format(datasets[0]), 'r') as f:
        y_train = np.array(json.load(f)).astype(float)
    y_train = y_train[:X_train.shape[0], ...]
    with open('{}/labels.json'.format(datasets[1]), 'r') as f:
        y_test = np.array(json.load(f)).astype(float)
    target_field = 0
    y_train = y_train[:, target_field]
    y_test = y_test[:, target_field]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    print('fitting..')
    clf = LogisticRegression(random_state=0,class_weight='balanced').fit(X_train, y_train)
    print('=====')
    print('Table3, Test Acc {:.4f}'.format(clf.score(X_test, y_test)))
    print('=====')

    X_train = np.load(save_path)
    X_test = np.load(human_save_path)

    X_train = X_train.reshape(X_train.shape[0], 10, -1)
    X_test = X_test.reshape(X_test.shape[0], 10, -1)
    X_train = X_train[:, 4:, :].reshape(X_train.shape[0], -1)
    X_test = X_test[:, 4:, :].reshape(X_test.shape[0], -1)

    with open('{}/labels.json'.format(datasets[0]), 'r') as f:
        y_train = np.array(json.load(f)).astype(float)
    y_train = y_train[:X_train.shape[0], ...]
    with open('{}/labels.json'.format(datasets[1]), 'r') as f:
        y_test = np.array(json.load(f)).astype(float)
    target_field = 0
    y_train = y_train[:, target_field]
    y_test = y_test[:, target_field]

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    print('fitting..')
    clf = LogisticRegression(random_state=0,class_weight='balanced').fit(X_train, y_train)
    print('=====')
    print('Table1, Test Acc {:.4f}'.format(clf.score(X_test, y_test)))
    print('=====')
