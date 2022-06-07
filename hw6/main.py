# ========================================================
#             Media and Cognition
#             Homework 6 Transformer
#             main.py: Transformer based scene text recognition
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
# ========================================================

# ==== import libs
from functools import total_ordering
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os
import cv2
import argparse

from network import CNNTransformer
from utils import dataLoader, LabelConverter, plot_loss_and_accuracies, visual_transformer_results


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
    #         please see ListDataset and dataLoader (line 19 and 89) in utils.py for details
    trainloader = dataLoader(train_im_dir, norm_height, norm_width, batch_size, training=True)
    valloader = dataLoader(val_im_dir, norm_height, norm_width, batch_size, training=False)

    # step 2: initialize the label converter
    #         please see LabelConverter (line 109) in utils.py for details
    label_converter = LabelConverter()

    # step 3: initialize the model
    model = CNNTransformer()
    model = model.to(device)
    if load_pretrain:
        try:
            checkpoint = torch.load(pretrain_path)
            model.load_state_dict(checkpoint['state_dict'])
            print(f'[Info] load pretrained model from {pretrain_path}')
        except Exception as e:
            print(f'[Warning] load pretrain model failed, the reason is:\n    {e}')
            print('[Warning] the model will be trained from scratch!')

    # step 4: define NLL(Negative Log Likelihood) loss function and optimizer
    # -- NLL loss function in PyTorch is nn.NLLLoss()
    #    please refer to the following document to look up its usage
    #    https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss
    criterion = nn.NLLLoss(ignore_index=2, reduction='mean')
    optimizer = optim.RMSprop(model.parameters(), lr) 

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
	
	# define the network forward process and loss computation
    # you may follow the below steps
    # 1. set model into training mode
    # 2. initialize a "total_loss" variable to sum up losses from each training step
    # 3. start to loop, fetch images and labels of each step from "trainloader"
    # 4.   convert label texts into tensors by "label_converter"
    # 5.   run the model forward process
    # 6.   compute loss by "criterion"
    # 7.   run the backward process and update model parameters
    # 8.   update "total_loss"
    # 9. return avg_loss, which is total_loss / len(trainloader)

    # hint: in general, we need log_probs and targets to compute NLL loss,
    #       "targets" can be obtained by "label_converter.encode(labels)",
    #       please refer to the following document for details:
    #       https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss
	#       
	#                           ###### NOTICE! ####### 
	#       you should shift the targets and the log_probs before computing the loss!
	#       you should change the shapes of the targets and the log_probs before computing the loss!
	

    # ==================================
    # TODO 7: complete train_one_epoch()
    # ==================================
    
    model.train()
    
    total_loss = 0.0 
    
    for step, (ims, words) in enumerate(trainloader):
        tgt, tgt_length = label_converter.encode(words)
        ims, tgt, tgt_length = ims.to(device), tgt.to(device), tgt_length.to(device)
        logits = model(ims, tgt, tgt_length)
        log_probs = nn.functional.log_softmax(logits, dim = 2)
        log_probs = log_probs[:-1, :, :].view(-1, log_probs.size(2))
        # 需要去掉log_probs的最后一行（为了与target的长度一致），并且将前两维压缩为一维
        target = tgt[1:, :].view(-1)
        # 需要去掉target的第一行，因为第一行是<sos>，解码时不会出现<sos>
        # log_probs 和 target的形状为[~, 32, 40]，第一个维度取决于label中最长的那个的长度 * 32
        loss = criterion(log_probs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(trainloader)


def val_one_epoch(model, valloader, label_converter, device):
    """
    validate the current model
    :param model: a model object
    :param valloader: validation data loader
    :param label_converter: label converter to decode model outputs into texts
    :param device: 'cpu' or 'cuda'
    :return: validation word accuracy
    """

    model.eval()
    n_correct = 0.
    n_total = 0.
    with torch.no_grad():
        for ims, labels in valloader:
            ims = ims.to(device)
            raw_preds, logits = model.inference(ims)
            # raw_preds: [30, 32]，每个时刻只输出预测概率最大的类别序号
            # logits: [29, 32, 40]，transformer的decoder的输出的logits的长度是由tgt_mask决定的
            preds = label_converter.decode(raw_preds)
            for pred, label in zip(preds, labels):
                n_correct += pred == label
            n_total += len(labels)

    return n_correct / n_total


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
    model = CNNTransformer()
    model = model.to(device)

    # step 2: load state_dict from saved model
    checkpoint = torch.load(model_path)
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
        raw_pred, logits = model.inference(x)
        pred = label_converter.decode(raw_pred)[0]
    print('prediction: {}\n'.format(pred))

    # auxiliary step: visualize probabilities output by Transformer
    savepath = os.path.splitext(im_path)[0] + '_vis.jpg'
    visual_transformer_results(im, logits, raw_pred, savepath)


if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 0
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
    parser.add_argument('--epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--model_save_epoch', type=int, default=10, help='the frequency of saving model')
    parser.add_argument('--load_pretrain', action='store_true', help='whether to load pretrained model')
    parser.add_argument('--pretrain_path', type=str, default='models/pretrain.pth',
                        help='path of the pretrained model')

    # configurations for prediction
    parser.add_argument('--model_path', type=str, default='models/pretrain.pth',
                        help='path of a saved model')
    parser.add_argument('--im_path', type=str, default='data/my_own/parking.png',
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
