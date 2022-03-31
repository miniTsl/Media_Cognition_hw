#========================================================
#             Media and Cognition
#             Homework 3 Convolutional Neural Network
#             test.py - test our model for character classification
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
#========================================================
import os 
import torch
import cv2
import string
import argparse
from train import dataLoader
from network import CNN

def test(data_file_path, ckpt_path, epoch, save_results, device='cpu'):
    '''
    The main testing procedure
    ----------------------------
    :param data_file_path: path to the file with training data
    :param ckpt_path: path to load checkpoints
    :param epoch: epoch of checkpoint you want to load
    :param save_results: whether to save results
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    '''

    if save_results:
        save_dir = os.path.join(ckpt_path, 'results')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    # construct testing data loader
    test_loader = dataLoader(data_file_path, norm_size=(32, 32), batch_size=1)

    
    print('[Info] loading checkpoint from %s ...'%os.path.join(ckpt_path, 'ckpt_epoch_%d.pth'%epoch))
    checkpoint = torch.load(os.path.join(ckpt_path, 'ckpt_epoch_%d.pth'%epoch))
    configs = checkpoint['configs']
    model = CNN(configs['in_channels'], configs['num_class'], batch_norm=configs['batch_norm'], p=configs['dropout'])
    # load model parameters (checkpoint['model_state']) we saved in model_path using model.load_state_dict()
    model.load_state_dict(checkpoint['model_state'])
    # put the model on CPU or GPU
    model = model.to(device)

    # enter the evaluation mode
    model.eval()
    correct = 0
    n = 0
    letters = string.ascii_letters[-26:]
    for input, label in test_loader:
        # set data type and device
        input, label = input.type(torch.float).to(device), label.type(torch.long).to(device)
        # get the prediction result
        pred = model(input)
        pred = torch.argmax(pred, dim = -1)
        label = label.squeeze(dim = 0)

        # set the name of saved images to 'idx_correct/wrong_label_pred.jpg'
        if pred == label:
            correct += 1
            save_name = '%04d_correct_%s_%s.jpg' % (n, letters[int(label)], letters[int(pred)])
        else:
            save_name = '%04d_wrong_%s_%s.jpg' % (n, letters[int(label)], letters[int(pred)])

        if save_results:
            cv2.imwrite(os.path.join(save_dir, save_name), 255*(input*0.5+0.5).squeeze(0).permute(1,2,0).detach().numpy())

        n += 1
    # calculate accuracy
    accuracy = float(correct) / float(len(test_loader))
    print('accuracy on the test set: %.3f'%accuracy)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # set configurations of the testing process
    parser.add_argument('--path', type=str, default='data/test', help='path to data file')
    parser.add_argument('--epoch', type=int, default=10, help='epoch of checkpoint you want to load')
    parser.add_argument('--ckpt_path', type=str, default='ckpt', help='path to load checkpoints')
    parser.add_argument('--save', action='store_true', default=False, help='whether to save results')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')

    opt = parser.parse_args()

    # run the testing procedure
    test(data_file_path=opt.path,
          ckpt_path=opt.ckpt_path,
          epoch=opt.epoch,
          save_results=opt.save,
          device=opt.device)