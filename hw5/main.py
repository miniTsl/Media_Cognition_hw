# ========================================================
#             Media and Cognition
#             Homework 5 Recurrent Neural Network
#             main.py: CNN-RNN-CTC based scene text recognition
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
# ========================================================

# ==== import libs
from functools import total_ordering
from sqlite3 import converters
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import cv2
import argparse

from network import CRNN
from utils import dataLoader, LabelConverter, plot_loss_and_accuracies, visual_ctc_results


# ==== training and validation

def train_val(train_im_dir='data/train', val_im_dir='data/train',  # data path configs
              norm_height=32, norm_width=128,  # image normalization configs
              n_epochs=20, batch_size=4, lr=1e-4,  # training configs
              model_save_epoch=5, model_save_dir='models',  # model saving configs
              load_pretrain=False, pretrain_path=None,  # pretrained model configs
              device='cpu'):
    '''
    The main training procedure
    ----------------------------
    :param train_im_dir: path to directory with training images and ground-truth file
    :param val_im_dir: path to directory with validation images and ground-truth file
    :param norm_height: image normalization height
    :param norm_width: image normalization width
    :param n_epochs: number of training epochs
    :param batch_size: training and validation batch size
    :param lr: learning rate
    :param model_save_epoch: save model after each {model_save_epoch} epochs
    :param model_save_dir: path to save the model
    :param load_pretrain: whether to load a pretrained model
    :param pretrain_path: path of the pretrained model
    :param device: 'cpu' or 'cuda'
    '''

    # step 1: initialize training and validation data loaders
    #         please see ListDataset and dataLoader (line 19 and line 92) in utils.py for details
    trainloader = dataLoader(train_im_dir, norm_height, norm_width, batch_size, training=True)
    valloader = dataLoader(val_im_dir, norm_height, norm_width, batch_size, training=False)

    # step 2: initialize the label converter
    #         please see LabelConverter (line 112) in utils.py for details
    label_converter = LabelConverter()  # 把文本转换为tensor

    # step 3: initialize the model
    model = CRNN()
    model = model.to(device)
    if load_pretrain:   # 加载预训练模型
        try:
            checkpoint = torch.load(pretrain_path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            print(f'[Info] load pretrained model from {pretrain_path}')
        except Exception as e:
            print(f'[Warning] load pretrain model failed, the reason is:\n    {e}')
            print('[Warning] the model will be trained from scratch!')

    # step 4: define CTC loss function and optimizer
    # -- CTC loss function in PyTorch is nn.CTCLoss()
    #    note that the first input of nn.CTCLoss() is logarithmized probabilities
    #    please refer to the following document to look up its usage
    #    https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss
    criterion = nn.CTCLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # step 5: training & validation

    # two lists to save training loss and validation accuracy for each epoch
    losses, accuracies = [], []

    for epoch in range(n_epochs):
        # train
        print('\nEpoch [{}/{}] start ...'.format(epoch + 1, n_epochs))
        train_loss = train_one_epoch(model, trainloader, optimizer, criterion, label_converter, device)
        losses.append(train_loss)

        # validation
        accuracy = val_one_epoch(model, valloader, label_converter, device)
        accuracies.append(accuracy)

        # show information of the epoch
        print('train loss = {:.3f}, validation word accuracy = {:.1f}%'
              .format(train_loss, 100 * accuracy))

        # save model
        if (epoch + 1) % model_save_epoch == 0:
            model_save_path = os.path.join(model_save_dir, 'model_epoch{}.pth'.format(epoch + 1))
            torch.save({'state_dict': model.state_dict()}, model_save_path)
            print('[Info] model saved in {}'.format(model_save_path))

    # draw the loss and accuracy curve
    plot_loss_and_accuracies(losses, accuracies)


def train_one_epoch(model, trainloader, optimizer, criterion, label_converter, device):
    """
    train the model for one epoch
    :param model: a model object
    :param trainloader: train data loader
    :param optimizer: a PyTorch optimizer
    :param criterion: loss function
    :param label_converter: label converter to encode texts into tensors
    :param device: 'cpu' or 'cuda'
    :return: the average training loss
    """
    # you may follow the below steps
    # 1. set model into training mode
    model.train()
    # 2. initialize a "total_loss" variable to sum up losses from each training step
    total_loss = 0.
    # 3. start to loop, fetch images and labels of each step from "trainloader"
    for ims, texts in trainloader:
    # 4.   convert label texts into tensors by "label_converter"
        targets, target_lengths = label_converter.encode(texts)
        ims = ims.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
    # 5.   run the model forward process
        logits, seq_lengths = model(ims)
    # 6.   compute loss by "criterion"
        loss = criterion(logits.log_softmax(2), targets, seq_lengths, target_lengths)
    # 7.   run the backward process and update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 8.   update "total_loss"
        total_loss += loss.item()
    # 9. return avg_loss, which is total_loss / len(trainloader)
    avg_loss = total_loss / len(trainloader)
    # hint: in general, we need log_probs, targets, input_lengths and target_lengths to compute CTC loss,
    #       "log_probs" can be transformed from "logits" output by model using "nn.functional.log_softmax(dim)",
    #       "targets" and "target_lengths" can be obtained by "label_converter.encode(labels)",
    #       "input_lengths" is output by model.
    #       please refer to the following document for details:
    #       https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html?highlight=ctcloss#torch.nn.CTCLoss

    # ==================================
    # TODO 3: complete train_one_epoch()
    # ==================================

    return avg_loss


def val_one_epoch(model, valloader, label_converter, device):
    """
    evaluate the current model
    :param model: a model object
    :param valloader: validation data loader
    :param label_converter: label converter to decode model outputs into texts
    :param device: 'cpu' or 'cuda'
    :return: validation word accuracy
    """

    # hint: you may follow the below steps
    # 1. set model into evaluation mode
    model.eval()
    # 2. initialize "n_correct" and "n_total" variables to save the numbers of correct and total images
    n_correct = 0.
    n_total = 0.

    # 3. loop under the no-gradient environment, fetch images and labels of each step from "valloader"
    with torch.no_grad():
        for ims, texts in valloader:
    # 4.   run model forward process and compute "logits"
            logits, _ = model(ims)
    # 5.   get raw predictions "raw_preds" by "logits.argmax(2)"
            raw_preds = logits.argmax(2)
    # 6.   use "label_converter.decode(raw_preds)" to obtain decoded texts
            preds = label_converter.decode(raw_preds)
    # 7.   update "n_total" and update "n_correct" by comparing decoded texts with labels
            n_total += len(texts)
            n_correct += sum([pred == text for pred, text in zip(preds, texts)]) 
    # 8. return accuracy, which is n_correct / n_total
    accuracy = n_correct/n_total
    # ================================
    # TODO 4: complete val_one_epoch()
    # ================================

    return accuracy


# ==== predict a new image using a trained model
def predict(model_path, im_path, norm_height=32, norm_width=128, device='cpu'):
    '''
    predict a new image using a trained model
    :param model_path: path of the saved model
    :param im_path: path of an image
    :param norm_height: image normalization height
    :param norm_width: image normalization width
    :param device: 'cpu' or 'cuda'
    '''

    # step 1: initialize a model and put it on device
    model = CRNN()
    model = model.to(device)

    # step 2: load state_dict from saved model
    checkpoint = torch.load(model_path, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    print('[Info] Load model from {}'.format(model_path))

    # step 3: initialize the label converter
    label_converter = LabelConverter()

    # step 4: read image and normalization
    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((norm_height, norm_width)),
                    transforms.ToTensor()])
    im = cv2.imread(im_path)
    if im is None:
        raise AssertionError(f'the image {im_path} may not exist, please check it.')
    x = transform(im)
    x = x.unsqueeze(0)  # add the batch dimension

    # step 5: run model
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
        raw_pred = logits.argmax(2)
        pred = label_converter.decode(raw_pred)[0]
    print('prediction: {}\n'.format(pred))

    # visualize probabilities output by CTC
    savepath = os.path.splitext(im_path)[0] + '_vis.jpg'
    visual_ctc_results(im, logits, savepath)


if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    # set configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or predict')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--norm_height', type=int, default=32, help='image normalization height')
    parser.add_argument('--norm_width', type=int, default=128, help='image normalization width')
    parser.add_argument('--epoch', type=int, default=40, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model_save_epoch', type=int, default=10, help='the frequency of saving model')
    parser.add_argument('--load_pretrain', action='store_true', help='whether to load pretrained model')
    parser.add_argument('--pretrain_path', type=str, default='models/pretrain.pth',
                        help='path of the pretrained model')

    # configurations for prediction
    parser.add_argument('--model_path', type=str, default='models/model_epoch40.pth',
                        help='path of a saved model')
    parser.add_argument('--im_path', type=str, default='data/my_own/a.png',
                        help='path of an image to be recognized')

    opt = parser.parse_args()

    # -- training and validation
    if opt.mode == 'train':
        train_val(
            train_im_dir='data/train',
            val_im_dir='data/validation',
            norm_height=opt.norm_height,
            norm_width=opt.norm_width,
            n_epochs=opt.epoch,
            batch_size=opt.batchsize,
            lr=opt.lr,
            model_save_epoch=opt.model_save_epoch,
            load_pretrain=opt.load_pretrain,
            pretrain_path=opt.pretrain_path,
            device=opt.device)

    # -- predict a new image
    elif opt.mode == 'predict':
        predict(
            model_path=opt.model_path,
            im_path=opt.im_path,
            norm_height=opt.norm_height,
            norm_width=opt.norm_width,
            device=opt.device)

    else:
        raise NotImplementedError(f'mode should be train or predict, but got {opt.mode}')
