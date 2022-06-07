# ========================================================
#             Media and Cognition
#             Homework 6 Transformer
#             utils.py: define data loader and functions for visualization
#             Tsinghua University
#             (C) Copyright 2022
# ========================================================
import string

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import cv2
import string
import matplotlib.pyplot as plt
import numpy as np
# Dataset and Data Loader
class ListDataset(Dataset):
    def __init__(self, im_dir, norm_height=32, norm_width=128, training=False):
        """
        :param im_dir: path to directory with images and ground-truth file
        :param norm_height: image normalization height
        :param norm_width: image normalization width
        :param training: bool, use data augmentation during training
        """

        # step 1: get image paths and labels from label file
        #         label file is "im_dir/gt.txt", each line in the file is "image_name label",
        with open(os.path.join(im_dir, 'gt.txt'), 'r') as f:
            # self.im_paths contains all image path
            # self.labels contains all ground-truth texts
            self.im_paths = []
            self.labels = []

            lines = f.readlines()
            for line in lines:
                im_name, label = line.split()
                self.im_paths.append(os.path.join(im_dir, im_name))
                self.labels.append(label)

        self.nSamples = len(self.im_paths)
        print(f'---- Finish loading {self.nSamples} samples from {im_dir} ----')

        # step 2: data augmentation and normalization
        if training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.0, 0.0, 0.0)],
                    p=0.5),
                transforms.RandomApply(
                    [transforms.RandomAffine(degrees=10.0,
                                             translate=(0.02, 0.05),
                                             shear=10.0)],
                    p=0.5),
                transforms.Resize((norm_height, norm_width)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((norm_height, norm_width)),
                transforms.ToTensor(),
            ])


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        """
        :param index: index of a sample
        :return: image in tensor format (3 channels) and label text
        """
        assert index <= len(self), 'index range error'

        # step 1: read an image
        im_path = self.im_paths[index]
        im = cv2.imread(im_path)

        # step 2: image pre-processing
        im = self.transform(im)  # ToTensor() will normalize images into [0, 1]
        im.sub_(0.5).div_(0.5)  # further normalize to [-1, 1]

        # step 3: get the label text
        # use label.lower() to get case-insensitive text
        label = self.labels[index].lower()
        return im, label


def dataLoader(im_dir, norm_height, norm_width, batch_size, training, workers=0, max_timestep=30):
    '''
    :param im_dir: path to directory with images and ground-truth file
    :param norm_height: image normalization height
    :param batch_size: batch size
    :param workers: number of workers for loading data in multiple threads
    :return: a data loader
    '''

    dataset = ListDataset(im_dir, norm_height, norm_width, training)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle= training,  # shuffle images only when training
                      num_workers=workers)

class LabelConverter:
    """
    A label converter is used to convert a label text into its corresponding tensor format.
    It can also convert the output of the model into predicted text.
    """
    def __init__(self):
        self.character_set =  string.ascii_lowercase + string.digits  # 26 letters and 10 digits
        self.transformer_dictionary = dict()
        self.transformer_reverse_dictionary = dict()
        self.transformer_reverse_dictionary[0] = '<sos>'  # for transformer, label 0 --> '<sos>', 1 --> '<eos>', 2 --> '<padding>', 3 --> 'unknown'
        self.transformer_reverse_dictionary[1] = '<eos>'
        self.transformer_reverse_dictionary[2] = '<pad>'
        self.transformer_reverse_dictionary[3] = '<unk>'
        for i, char in enumerate(self.character_set):
            self.transformer_dictionary[char] = i + 4
            self.transformer_reverse_dictionary[i + 4] = char


    def encode(self, words):
        """
        Encoding a list of words into PyTorch Tensors to fit the input of Transformer
        :param words: list of "batchsize" words
        :return targets: torch.LongTensor with size [sum(target_lengths)], all the targets
        :return target_lengths: torch.LongTensor with size [batchsize], length of each word
        """
        batch_size = len(words)
        tgt_lengths = [len(label) for label in words]
  
        padded_tgt = np.zeros(
            (max(tgt_lengths) + 2, batch_size))
        for i, label in enumerate(words):
            padded_tgt[0, i] = 0    # 添加<sos>
            padded_tgt[1:len(label) + 1, i] = [int(self.transformer_dictionary[char]) for _, char in enumerate(label)]
            padded_tgt[len(label) + 1, i] = 1   # 添加<eos>
            padded_tgt[len(label) + 2:, i] = 2  # 添加<pad>

        return torch.LongTensor(padded_tgt), torch.LongTensor(tgt_lengths)

    def decode(self, raw_preds):
        """
        Transformer decoding method
        :param raw_preds: torch.LongTensor of size [w, b],
                          raw_preds contains <unk> and blank symbols.
                          w is the length of feature sequences and b is batchsize,
        :return: a list of prediction texts
        """
        raw_preds = raw_preds.permute(1, 0).cpu().numpy()
        preds = []
        for raw_pred in raw_preds:
            raw_pred = raw_pred.tolist()  # a list of length [w]

            # step 1: merge repeated characters between special symbols, break when meet <eos>
            merged = []
            prev_char = ''
            real_prev_char = ''
            for i, char in enumerate(raw_pred):
                if(char == 1): #<eos>
                    break
                if char != prev_char and char > 3:  # ignore special symbols
                    merged.append(char)
                elif char == prev_char and char > 3 and real_prev_char > 3:
                    merged.append(char)
                if(char > 3):
                    prev_char = char
                real_prev_char = char

            # step2: remove all special symbols
            pred = []
            for char in merged:
                if char < 4: # ignore <sos><eos><pad><unk>
                    continue
                pred.append(char)
            preds.append(''.join([self.character_set[idx - 4] for idx in pred]))

        return preds


def plot_loss_and_accuracies(losses, accuracies, savepath='loss_and_accuracy.jpg'):
    """
    :param losses: list of losses for each epoch
    :param accuracies: list of accuracies for each epoch
    :param savepath: path to save figure
    """

    # draw loss
    ax = plt.subplot(2, 1, 1)
    ax.plot(losses)
    # set labels
    ax.set_xlabel('training epoch')
    ax.set_ylabel('training loss')

    # draw accuracy
    ax = plt.subplot(2, 1, 2)
    ax.plot(accuracies)
    # set labels
    ax.set_xlabel('training epoch')
    ax.set_ylabel('validation accuracy')

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    print(f'loss and accuracy curves has been saved in {savepath}')


def visual_transformer_results(image, logits, raw_pred, savepath='visualization.jpg'):
    """
    visualize the model's classification sequence, we can see the alignment learnt by Transformer
    :param image: the original image
    :param logits: logits output by model
    :param savepath: path to save figure
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [1, 5]})

    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), aspect='auto')
    ax1.axis('off')

    probs = logits.softmax(2)[:, 0, :].transpose(1, 0).cpu().numpy()
    for i in range(len(raw_pred)):
        if(raw_pred[i] == 1):
            pred_length = i
            break
    

    im_probs = ax2.imshow(probs[:, :pred_length], aspect='auto')
    plt.xlabel('Time step')
    plt.yticks(list(range(logits.size(2))),
               ['<sos>','<eos>','<pad>', '<unk>'] + list(string.ascii_lowercase + string.digits))

    plt.subplots_adjust(bottom=0.12, top=0.97, left=0.1, right=0.95, hspace=0.05)
    cb_ax = fig.add_axes([0.1, 0.05, 0.85, 0.01])
    fig.colorbar(im_probs, cax=cb_ax, orientation="horizontal")

    plt.savefig(savepath, dpi=300)
    print(f'Transformer visualization has been saved as {savepath}')
