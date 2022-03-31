#========================================================
#             Media and Cognition
#             Homework 3 Convolutional Neural Network
#             cal_entropy.py - calculate entropy of images and features
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
#========================================================

import argparse
import cv2
import os
import numpy as np
import torch

from train import dataLoader
from network import CNN


def im_entropy(im):
    """
    Calculate entropy of one single image
    :param im: a greyscale image in numpy format
    :return: entropy of the image
    """

    h, w = im.shape
    hist = np.histogram(im.reshape(-1), bins=256)[0]
    probs = hist / (h * w)
    probs = probs[probs > 0]
    ent = np.sum(-probs * np.log(probs))
    return ent


def im_entropy_dataset(file_path, norm_size):
    """
    Calculate the average entropy of images in a dataset
    :param file_path: path to directory with images
    :param norm_size: image normalization size, (width, height)
    :return: the average entropy
    """

    dataloader = dataLoader(file_path, norm_size, batch_size=1)

    ent = 0.
    # calculate entropy of each image
    for im, _ in dataloader:
        im = im.squeeze()
        im = (im * 255 + 255) / 2
        ent += im_entropy(im.cpu().numpy())

    return ent / len(dataloader)


def label_entropy_random(n_class=26):
    """
    We randomly guess a label for each input.
    :param n_class: number of class
    :return: the entropy
    """
    return -np.log(1 / n_class)


def label_entropy_statistics(file_path):
    """
    We use the statistics results for prediction.
    :param file_path: json file containing image names and labels
    :return: the entropy
    """

    dataloader = dataLoader(file_path, (32, 32), batch_size=1)
    # convert labels to int numbers
    labels = [label.cpu().item() for _, label in dataloader]

    # calculate entropy
    hist = np.histogram(np.array(labels), bins=26)[0]
    probs = hist / len(labels)
    probs = probs[probs > 0]
    ent = np.sum(-probs * np.log(probs))

    return ent


def label_entropy_model(file_path, norm_size, batch_size, model_path, device='cpu'):
    """
    We use the trained model for prediction.
    :param file_path: path to directory with images
    :param norm_size: image normalization size, (width, height)
    :param batch_size: batch size
    :param model_path: path of the saved model
    :param device: 'cpu' or 'cuda'
    :return: the entropy
    """

    # initialize dataloader and model
    dataloader = dataLoader(file_path, norm_size, batch_size)
    print('[Info] loading checkpoints from %s ...'% model_path)
    checkpoint = torch.load(model_path)
    configs = checkpoint['configs']
    model = CNN(configs['in_channels'], configs['num_class'], batch_norm=configs['batch_norm'], p=configs['dropout'])
    # load model parameters (checkpoint['model_state']) we saved in model_path using model.load_state_dict()
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)

    # extract features
    outs = []
    model.eval()
    with torch.no_grad():
        for ims, _ in dataloader:
            ims = ims.to(device)
            out = model(ims)
            outs.append(out)

    # calculate entropy
    probs = torch.cat(outs, 0).softmax(1)  # [n_ims, 26], probabilities of predicted characters
    probs = probs.cpu().numpy()
    ent = 0.
    for prob in probs:
        prob = prob[prob > 0]
        ent -= np.sum(prob * np.log(prob))
    ent /= len(probs)

    return ent


def feature_entropy_channel(x, n_bins, ignore_lowest):
    """ The entropy of a single channel feature map
    :param x: feature map with shape [h, w] in pytorch tensor form
    :param n_bins: the bins to be divided
    :param ignore_lowest: whether to ignore the lowest value when calculating the entropy
    :return: the entropy
    """

    x = x.view(-1)
    if ignore_lowest:
        assert x.max() > x.min(), 'the feature map is identical, cannot ignore the lowest value'
        x = x[x > x.min()]

    hist = np.histogram(x.cpu().numpy(), bins=n_bins)[0]
    probs = hist / len(x)
    probs = probs[probs > 0]
    ent = np.sum(-probs * np.log(probs))

    return ent


def feature_entropy(x, n_bins, ignore_lowest, reduction='mean'):
    """ The entropy of feature maps
    :param x: feature map with shape [c, h, w] in pytorch tensor form
    :param n_bins: the bins to be divided
    :param ignore_lowest: whether to ignore the lowest value when calculating the entropy
    :param reduction: 'mean' or 'sum', the way to reduce results of c channels
    :return: the entropy
    """

    ent = 0.
    for f in x:
        if ignore_lowest and f.max() == f.min():
            continue
        ent += feature_entropy_channel(f, n_bins, ignore_lowest)

    assert reduction in ['mean', 'sum'], 'reduction should be mean or sum'
    if reduction == 'mean':
        ent /= x.size(0)
    return ent


def feature_entropy_dataset(n_bins, ignore_lowest, reduction, file_path,
                            norm_size, batch_size, model_path, device='cpu'):
    """
    Calculate entropy of features extracted by our model.
    :param n_bins: the bins to be divided
    :param ignore_lowest: whether to ignore the lowest value when calculating the entropy
    :param reduction: 'mean' or 'sum', the way to reduce results of c channels
    :param file_path: path to directory with images
    :param norm_size: image normalization size, (width, height)
    :param batch_size: batch size
    :param model_path: path of the saved model
    :param device: 'cpu' or 'cuda'
    :return: the entropy of features
    """

    # initialize dataloader and model
    dataloader = dataLoader(file_path, norm_size, batch_size)
    print('[Info] loading checkpoints from %s ...'% model_path)
    checkpoint = torch.load(model_path)
    configs = checkpoint['configs']
    model = CNN(configs['in_channels'], configs['num_class'], batch_norm=configs['batch_norm'], p=configs['dropout'])
    # load model parameters (checkpoint['model_state']) we saved in model_path using model.load_state_dict()
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)

    # extract features and calculate entropy
    ent1, ent2, ent3, ent4, ent5 = 0., 0., 0., 0., 0.
    n_ims = 0
    model.eval()
    with torch.no_grad():
        for ims, _ in dataloader:
            ims = ims.to(device)
            feats1, feats2, feats3, feats4, feats5, _ = model(ims, return_features=True)
            n_ims += ims.size(0)
            for f1, f2, f3, f4, f5 in zip(feats1, feats2, feats3, feats4, feats5):
                ent1 += feature_entropy(f1, n_bins, ignore_lowest, reduction)
                ent2 += feature_entropy(f2, n_bins, ignore_lowest, reduction)
                ent3 += feature_entropy(f3, n_bins, ignore_lowest, reduction)
                ent4 += feature_entropy(f4, n_bins, ignore_lowest, reduction)
                ent5 += feature_entropy(f5, n_bins, ignore_lowest, reduction)

    return ent1 / n_ims, ent2 / n_ims, ent3 / n_ims, ent4 / n_ims, ent5 / n_ims


def entropy_single_input(im_path, norm_size, model_path,
                         n_bins, ignore_lowest, reduction,
                         device='cpu'):
    """
    Calculate entropy of a single image and its prediction
    :param im_path: path to an image file
    :param norm_size: image normalization size, (width, height)
    :param model_path: path of the saved model
    :param n_bins: the bins to be divided
    :param ignore_lowest: whether to ignore the lowest value when calculating the entropy
    :param reduction: 'mean' or 'sum', the way to reduce results of c channels
    :param device: 'cpu' or 'cuda'
    :return: image entropy and predicted probability entropy
    """

    # read image and calculate image entropy
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im, norm_size)
    ent_im = im_entropy(im)
    # preprocess
    im = (torch.from_numpy(im).float() - 127.5) / 127.5
    im = im.view(1, 1, norm_size[1], norm_size[0])
    im = im.to(device)

    # initialize the model
    print('[Info] loading checkpoints from %s ...'% model_path)
    checkpoint = torch.load(model_path)
    configs = checkpoint['configs']
    model = CNN(configs['in_channels'], configs['num_class'], batch_norm=configs['batch_norm'], p=configs['dropout'])
    # load model parameters (checkpoint['model_state']) we saved in model_path using model.load_state_dict()
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)

    # calculate prediction entropy
    model.eval()
    with torch.no_grad():
        f1, f2, f3, f4, f5, out = model(im, return_features=True)
    ent_f1 = feature_entropy(f1[0], n_bins, ignore_lowest, reduction)
    ent_f2 = feature_entropy(f2[0], n_bins, ignore_lowest, reduction)
    ent_f3 = feature_entropy(f3[0], n_bins, ignore_lowest, reduction)
    ent_f4 = feature_entropy(f4[0], n_bins, ignore_lowest, reduction)
    ent_f5 = feature_entropy(f5[0], n_bins, ignore_lowest, reduction)
    pred = out[0].argmax().item()
    pred = chr(pred + ord('A'))
    prob = out[0].softmax(0).cpu().numpy()
    confidence = prob.max()
    prob = prob[prob > 0]
    ent_pred = np.sum(-prob * np.log(prob))

    return pred, confidence, ent_im, ent_f1, ent_f2, ent_f3, ent_f4, ent_f5, ent_pred, f1[0], f2[0], f3[0], f4[0], f5[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='dataset',
                        help='if mode is dataset, then calculate average entropy of the whole dataset; \
                              if mode is single, then calculate the entropy of a single image')
    parser.add_argument('--train_file_path', type=str, default='data/train',
                        help='file list of training image paths and labels')
    parser.add_argument('--test_file_path', type=str, default='data/test',
                        help='file list of test image paths and labels')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--ckpt_path', type=str, default='bn_ckpt',
                        help='path of a saved model')
    parser.add_argument('--epoch', type=int, default=10, help='epoch of checkpoint you want to load')
    parser.add_argument('--im_path', type=str, default='', help='path of an image file')

    opt = parser.parse_args()

    w, h = 32, 32
    model_path = os.path.join(opt.ckpt_path, 'ckpt_epoch_%d.pth' % opt.epoch)
    if opt.mode == 'dataset':
        print('\ncalculating entropy of dataset...')

        # image entropy
        ent_im = im_entropy_dataset(opt.test_file_path, (w, h))
        # feature entropy
        ent_f1, ent_f2, ent_f3, ent_f4, ent_f5 = feature_entropy_dataset(256, False,
                                                         'mean', 
                                                         opt.test_file_path, (w, h),
                                                         8, model_path,
                                                         opt.device)
        # label entropy
        ent_rand = label_entropy_random()
        ent_statistics = label_entropy_statistics(opt.train_file_path)
        ent_model = label_entropy_model(opt.test_file_path, (w, h),
                                        8, model_path, opt.device)
        
        print('Entropy of input test images = {:.2f}'.format(ent_im))
        print('Entropy of features = {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(ent_f1, ent_f2, ent_f3, ent_f4, ent_f5))
        print('Entropy of random guess = {:.2f}'.format(ent_rand))
        print('Entropy of symbols in text labels = {:.2f}'.format(ent_statistics))
        print('Entropy of using trained model = {:.2f}'.format(ent_model))

    elif opt.mode == 'single':
        print('\n{}:'.format(os.path.basename(opt.im_path)))
        pred, confidence, ent_im, ent_f1, ent_f2, ent_f3, ent_f4, ent_f5, ent_pred, f1, f2, f3, f4, f5 = \
            entropy_single_input(opt.im_path, (w, h), model_path, 256,
                                 False, 'mean', opt.device)
        print('Recognition result: {} (confidence = {:.2f})'.format(pred, confidence))
        print('Entropy of input image = {:.2f}'.format(ent_im))
        print('Entropy of features = {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(ent_f1, ent_f2, ent_f3, ent_f4, ent_f5))
        print('Entropy of prediction = {:.2f}'.format(ent_pred))

    else:
        raise NotImplementedError('mode should be dataset or single')
