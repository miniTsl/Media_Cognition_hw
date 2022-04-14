# ========================================================
#             Media and Cognition
#             Homework 4 Support Vector Machine
#             classify_hw.py - Character/Background classification
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2021
# ========================================================

# ==== Part 0: import libs
from statistics import mode
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from svm_hw import libsvm, SVM_HINGE

# argparse is used to conveniently set our configurations
import argparse


# ==== Part 1: data loader

# construct a dataset and a data loader, more details can be found in
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataloader


class FeatureDataset(Dataset):

    def __init__(self, file_path):
        """
        :param file_path: .npy file of the data
        """
        # this time we will try to classify an image into character / background
        # the 2D features of the images have been already extracted by a CNN and saved as npy files

        self.data = np.load(file_path)
        self.labels = np.concatenate([np.ones(self.data.shape[0] // 2).astype(np.float32),
                                      -1.0 * np.ones(self.data.shape[0] // 2).astype(np.float32)], axis=0)

    # TODO: define the __len__ function of the Dataset class
    # return the number of samples (N) in self.data, using .shape[dim]
    # the shape of self.data is (N, channels)
    def __len__(self):
        return self.data.shape[0]

    # TODO: define the __getitem__ function of the Dataset class
    # item is an integer >= 0, indicating the index of an sample
    # return the corresponding data and label according to item
    # the returned feature should be of the shape (channels, ) and the returned label should be of the shape (1, )
    def __getitem__(self, item):
        # feature denotes one element in self.data, and label denotes one element in self.labels
        feature = self.data[item]
        label = self.labels[item]
        return feature, label


def dataLoader(file_path, batch_size, workers=0):
    """
    :param file_path: file with image paths and labels
    :param batch_size: batch size
    :param workers: number of workers for loading data in multiple threads
    :return: a data loader
    """

    dataset = FeatureDataset(file_path)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True if 'train' in file_path else False,  # shuffle images only when training
                      num_workers=workers)


# ==== Part 2: training and validation of the hinge loss version SVM

def train_val_hinge(train_file_path, val_file_path,
                    feature_channels, C, n_epochs, batch_size,
                    lr, valInterval, device='cpu'):
    '''
    The main training procedure
    ----------------------------
    :param train_file_path: npy file of training features
    :param val_file_path: npy file of validation features
    :param feature_channels: the channels of the feature
    :param C: the relaxation factor of the SVM
    :param n_epochs: number of training epochs
    :param batch_size: batch size of training and validation
    :param lr: learning rate
    :param valInterval: the frequency of validation, e.g., if valInterval = 5, then do validation after each 5 training epochs
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    '''

    # TODO: training and validation data loader using the previous self-defined function dataLoader()
    trainloader = dataLoader(train_file_path, batch_size)
    valloader = dataLoader(val_file_path, batch_size)

    C = C * len(trainloader)

    # TODO: initialize the hinge-loss type SVM;the SVM_HINGE class needs two parameters: in_channels, and C.
    model = SVM_HINGE(feature_channels, C)
    # TODO: put the model on CPU or GPU
    model = model.to(device)
    # TODO: initialize the Adam optimizer with model parameters and learning rate
    optimizer = optim.Adam(model.parameters(), lr)

    # training
    # to save loss of each training epoch in a python "list" data structure
    losses = []

    for epoch in range(n_epochs):
        # TODO: set the model in training mode
        model.train()
        # to save total loss in one epoch
        total_loss = 0.
        # TODO: get a batch of data; you may need enumerate() to iteratively get data from trainloader.
        # you can refer to previous homework, for example hw2
        for idx, (feas, labels) in enumerate(trainloader):
            # TODO: set data type (.float()) and device (.to())
            feas, labels = feas.float().to(device), labels.float().to(device)
            # TODO: clear gradients in the optimizer
            optimizer.zero_grad()
            # TODO: run the model with hinge loss; the model needs two inputs: feas and labels
            out, loss = model(feas, labels)
            # TODO: back-propagation on the computation graph
            loss.backward()
            # sum up of total loss, loss.item() return the value of the tensor as a standard python number
            # this operation is not differentiable
            total_loss += loss.item()
            # TODO: call a function to update the parameters of the models
            optimizer.step()

        # average of the total loss for iterations
        avg_loss = total_loss / len(trainloader)
        losses.append(avg_loss)
        print('Epoch {:02d}: loss = {:.3f}'.format(epoch + 1, avg_loss))

        # validation
        if (epoch + 1) % valInterval == 0:
            # TODO: set the model in evaluation mode
            model.eval()
            n_correct = 0.  # number of images that are correctly classified
            n_feas = 0.  # number of total images
            with torch.no_grad():  # we do not need to compute gradients during validation
                # TODO: inference on the validation dataset, similar to the training stage but use valloader.
                for idx, (feas, labels) in enumerate(valloader):
                    # TODO: set data type (.float()) and device (.to())
                    feas, labels = feas.float().to(device), labels.float().to(device)
                    # TODO: run the model; at the validation step, the model only needs one input: feas
                    # _ refers to a placeholder, which means we do not need the second returned value during validating
                    out, _ = model(feas)
                    predictions = out[:, 0]
                    # sum up the number of images correctly recognized
                    n_correct += torch.sum((predictions == labels).float())
                    # sum up the total image number
                    n_feas += feas.size(0)

            # show prediction accuracy
            print('Epoch {:02d}: validation accuracy = {:.1f}%'.format(epoch + 1, 100 * n_correct / n_feas))

    # save model parameters in a file
    model_save_path = 'saved_models/recognition.pth'.format(epoch + 1)
    torch.save({'state_dict': model.state_dict(),
                'configs': {
                    'feature_channels': feature_channels,
                    'C': C}
                }, model_save_path)
    print('Model saved in {}\n'.format(model_save_path))

    W = model.W.data.cpu().numpy()
    b = model.b.data.cpu().numpy()
    train_data = np.load(train_file_path)
    train_labels = np.concatenate([np.ones(train_data.shape[0] // 2).astype(np.float32),
                                   -1.0 * np.ones(train_data.shape[0] // 2).astype(np.float32)], axis=0)

    sv = np.argwhere((((W * train_data).sum(1) + np.reshape(b, (1, ))) * train_labels <= 1.0).astype(np.int32) == 1)
    sv = sv[:, 0].tolist()
    # draw the loss curve
    plot_loss(losses)
    return W, b, sv


# ==== Part 3: draw the loss curve
def plot_loss(losses):
    '''
    :param losses: list of losses for each epoch
    :return:
    '''

    f, ax = plt.subplots()

    # draw loss
    ax.plot(losses)

    # set labels
    ax.set_xlabel('training epoch')
    ax.set_ylabel('loss')

    # show the plots
    plt.show()


# ==== Part 4: draw the feature distribution and the classification boundary
def plot_feature(train_features, val_features, train_labels, val_labels, sv, W, b):
    train_labels = (train_labels > 0.0).astype(np.int32)
    val_labels = (val_labels > 0.0).astype(np.int32)
    train_labels[sv] = 2
    foreground = list(set([i for i in range(train_labels.shape[0] // 2)]) - set(sv))
    foreground_sv = list(set([i for i in range(train_labels.shape[0] // 2)]) - set(foreground))
    background = list(set([i + train_labels.shape[0] // 2 for i in range(train_labels.shape[0] // 2)]) - set(sv))
    background_sv = list(set([i + train_labels.shape[0] // 2 for i in range(train_labels.shape[0] // 2)]) - set(background))
    f, ax = plt.subplots()
    ax.scatter(train_features[foreground, 0], train_features[foreground, 1], marker='.', c='r')
    ax.scatter(train_features[foreground_sv, 0], train_features[foreground_sv, 1], marker='.', c='b')
    ax.scatter(train_features[background, 0], train_features[background, 1], marker='x', c='g')
    ax.scatter(train_features[background_sv, 0], train_features[background_sv, 1], marker='x', c='b')
    x = np.linspace(0,0.6,100)
    ax.plot(x, -W[0,0]/W[0,1] * x - b/W[0,1], c='y')
    plt.show()
    f, ax = plt.subplots()
    foreground_val = [i for i in range(val_labels.shape[0] // 2)]
    background_val = [i + val_labels.shape[0] // 2 for i in range(val_labels.shape[0] // 2)]
    ax.scatter(val_features[foreground_val, 0], val_features[foreground_val, 1], marker='.', c='r')
    ax.scatter(val_features[background_val, 0], val_features[background_val, 1], marker='x', c='b')
    x = np.linspace(0, 0.6, 100)
    ax.plot(x, -W[0, 0] / W[0, 1] * x - b / W[0, 1], c='y')
    plt.show()


if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # set configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='hinge', help='hinge, baseline')
    parser.add_argument('--train_file_path', type=str, default='data/train.npy',
                        help='file list of training image paths and labels')
    parser.add_argument('--val_file_path', type=str, default='data/val.npy',
                        help='file list of validation image paths and labels')
    parser.add_argument('--batchsize', type=int, default=2400, help='batch size')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')

    # configurations for training
    parser.add_argument('--epoch', type=int, default=200, help='number of training epochs')
    parser.add_argument('--valInterval', type=int, default=10, help='the frequency of validation')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--C', type=float, default=0.1, help='the relaxation factor in hinge loss')
    # configurations for test and prediction
    parser.add_argument('--model_path', type=str, default='saved_models/recognition.pth', help='path of a saved model')

    opt = parser.parse_args()

    # -- run the code for training and validation
    if opt.mode == 'hinge':
        W, b, sv = train_val_hinge(train_file_path=opt.train_file_path,
                        val_file_path=opt.val_file_path,
                        feature_channels=2,
                        C=opt.C,
                        n_epochs=opt.epoch,
                        batch_size=opt.batchsize,
                        lr=opt.lr,
                        valInterval=opt.valInterval,
                        device=opt.device)

    elif opt.mode == 'baseline':
        W, b, sv = libsvm(opt.train_file_path, opt.val_file_path, opt.C)

    else:
        print('mode should be train, test, or predict')
        raise NotImplementedError
    train_data = np.load(opt.train_file_path)
    train_labels = np.concatenate([np.ones(train_data.shape[0] // 2).astype(np.float32),
                                   -1.0 * np.ones(train_data.shape[0] // 2).astype(np.float32)], axis=0)
    val_data = np.load(opt.val_file_path)
    val_labels = np.concatenate([np.ones(val_data.shape[0] // 2).astype(np.float32),
                                 -1.0 * np.ones(val_data.shape[0] // 2).astype(np.float32)], axis=0)
    plot_feature(train_features=train_data, val_features=val_data, train_labels=train_labels, val_labels=val_labels,
                 sv=sv, W=W, b=b)


