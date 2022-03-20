#========================================================
#             Media and Cognition
#             Homework 2 Multilayer Perceptron
#             classification.py - character classification
#             Student ID:
#             Name:
#             Tsinghua University
#             (C) Copyright 2022
#========================================================

# ==== Part 0: import libs
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import json, cv2, os, string
import matplotlib.pyplot as plt

# this time we implement our networks and loss functions in other python scripts, and import them here
from network import MLP
from losses import CrossEntropyLoss

# argparse is used to conveniently set our configurations
import argparse
from sklearn.manifold import TSNE


# ==== Part 1: data loader
# construct a dataset and a data loader, more details can be found in
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html?highlight=dataloader
class ListDataset(Dataset):
    def __init__(self, im_dir, file_path, norm_size=(32, 32)):
        '''
        :param im_dir: path to directory with images
        :param file_path: json file containing image names and labels
        :param norm_size: image normalization size, (height, width)
        '''

        # this time we will try to recognize 26 English letters (case-insensitive)
        letters = string.ascii_letters[-26:]  # ABCD...XYZ
        self.alphabet = {letters[i]:i for i in range(len(letters))}
        # you can print self.alphabet to see what it is:字典内部是键值对
        # print(self.alphabet)

        # get image paths and labels from json file
        with open(file_path, 'r') as f:
            imgs = json.load(f)
            im_names = list(imgs.keys())

            self.im_paths = [os.path.join(im_dir, im_name) for im_name in im_names]
            self.labels = list(imgs.values())

            
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(norm_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        #read an image and convert it to grey scale
        im_path = self.im_paths[index]
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # image pre-processing, after pre-processing, the values of image pixels are converted to [0,1]
        im = self.transform(im)
        # convert values to [-1, 1]
        im.sub_(0.5).div_(0.5)

        # get the label of the current image
        # upper() is used to convert a letter into uppercase
        label = self.labels[index].upper()

        # convert an English letter into a <number index>
        label = self.alphabet[label]

        return im, label


def dataLoader(im_dir, file_path, norm_size, batch_size, workers=0):
    '''
    :param im_dir: path to directory with images
    :param file_path: file with image paths and labels
    :param norm_size: image normalization size, (height, width)
    :param batch_size: batch size
    :param workers: number of workers for loading data in multiple threads
    :return: a data loader
    '''

    dataset = ListDataset(im_dir, file_path, norm_size)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True if 'train' in file_path else False,  # shuffle images only when training
                      num_workers=workers)


# ==== Part 3: training and validation
def train_val(im_dir, train_file_path, val_file_path,
              hidden_size, n_layers, act_type,
              norm_size, n_epochs, batch_size, n_letters,
              lr, optim_type, momentum, weight_decay,
              valInterval, device='cpu'):
    '''
    The main training procedure
    ----------------------------
    :param im_dir: path to directory with images
    :param train_file_path: file list of training image paths and labels
    :param val_file_path: file list of validation image paths and labels
    :param hidden_size: a list of hidden size for each hidden layer
    :param n_layers: number of layers in the MLP
    :param act_type: type of activation function, can be none, sigmoid, tanh, or relu
    :param norm_size: image normalization size, (height, width)
    :param n_epochs: number of training epochs
    :param batch_size: batch size of training and validation
    :param n_letters: number of classes, in this task it is 26 English letters
    :param lr: learning rate
    :param optim_type: optimizer, can be 'sgd', 'adagrad', 'rmsprop', 'adam', or 'adadelta'
    :param momentum: only used if optim_type == 'sgd'
    :param weight_decay: the factor of L2 penalty on network weights
    :param valInterval: the frequency of validation, e.g., if valInterval = 5, then do validation after each 5 training epochs
    :param device: 'cpu' or 'cuda', we can use 'cpu' for our homework if GPU with cuda support is not available
    '''

    # training and validation data loader
    trainloader = dataLoader(im_dir, train_file_path, norm_size, batch_size)
    valloader = dataLoader(im_dir, val_file_path, norm_size, batch_size)

    # TODO 1: initialize the MLP model and loss function
    # what is the input size of the MLP?
    # hint 1: we convert an image to a vector as the input of the MLP, 
    # each image has shape [norm_size[0], norm_size[1]]
    # hint 2: Input parameters for MLP: input_size, output_size, hidden_size, n_layers, act_type
    input_size = norm_size[0]*norm_size[1] # 输入数据的形状是norm数组拉平后的长度
    model  = MLP(
        input_size = input_size, 
        output_size = n_letters,
        hidden_size=hidden_size,
        n_layers = n_layers,
        act_type=act_type
        )
    # loss function
    myloss = CrossEntropyLoss.apply
    # End TODO 1
    # put the model on CPU or GPU
    model = model.to(device)

    # optimizer
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    elif optim_type == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr, weight_decay=weight_decay)
    elif optim_type == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr, weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    elif optim_type == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr, weight_decay=weight_decay)
    else:
        print('[Error] optim_type should be one of sgd, adagrad, rmsprop, adam, or adadelta')
        raise NotImplementedError

    # training
    # to save loss of each training epoch in a python "list" data structure
    losses = []
    for epoch in range(n_epochs):
        # set the model in training mode
        model.train()
        # to save total loss in one epoch
        total_loss = 0.
        #TODO 2: calculate losses and train the network using the optimizer
        for step, (ims, labels) in enumerate(trainloader):  # get a batch of data

            # step 1: set data type and device
            ims, labels = ims.to(device), labels.to(device)
            # step 2: convert an image to a vector as the input of the MLP
            input = ims.view(ims.size(0),-1)
            # hint: clear gradients in the optimizer
            optimizer.zero_grad()
            # step 3: run the model which is the forward process
            logits = model(input)
            # step 4: compute the loss, and call backward propagation function
            loss = myloss(logits, labels)
            loss.backward()
            # step 5: sum up of total loss, loss.item() return the value of the tensor as a standard python number
            # this operation is not differentiable
            total_loss += loss.item()
            # step 6: call a function, optimizer.step(), to update the parameters of the model
            optimizer.step()
            # End TODO 2

        # average of the total loss for iterations
        # 打印各个epoch的平均误差
        avg_loss = total_loss / len(trainloader)
        losses.append(avg_loss)
        print('Epoch {:02d}: loss = {:.3f}'.format(epoch + 1, avg_loss))

        # validation在每隔valInterval个epoch后验证一下子
        if (epoch + 1) % valInterval == 0:

            # set the model in evaluation mode
            model.eval()

            n_correct = 0.  # number of images that are correctly classified
            n_ims = 0.  # number of total images

            with torch.no_grad():  # we do not need to compute gradients during validation
                # calculate losses for validation data and do not need train the network
                for ims, labels in valloader:
                    # set data type and device
                    ims, labels = ims.to(device), labels.type(torch.float).to(device)
                    # convert an image to a vector as the input of the MLP
                    input = ims.view(ims.size(0), -1)
                    # run the model which is the forward process
                    out = model(input)
                    # get the predicted value by the output using out.argmax(1)
                    predictions = out.argmax(1)
                    # sum up the number of images correctly recognized and the total image number
                    # 需要把逻辑判断的结果转变为float类型？
                    n_correct += torch.sum((predictions == labels).float())
                    n_ims += ims.size(0)
            # show prediction accuracy
            print('Epoch {:02d}: validation accuracy = {:.1f}%'.format(epoch + 1, 100 * n_correct / n_ims))

    # save model parameters in a file
    model_save_path = 'saved_models/recognition.pth'.format(epoch + 1)

    torch.save({'state_dict': model.state_dict(),
                'configs': {
                    'norm_size': norm_size,
                    'output_size': n_letters,
                    'hidden_size': hidden_size,
                    'n_layers': n_layers,
                    'act_type': act_type}
                }, model_save_path)
    print('Model saved in {}\n'.format(model_save_path))

    # draw the loss curve
    plot_loss(losses)


# ==== Part 4: test
def test(model_path, im_dir='data/character_classification/images',
         test_file_path='data/character_classification/test.json',
         batch_size=8, device='cpu'):
    '''
    Test and visualize procedure
    ---------------
    :param model_path: path of the saved model
    :param im_dir: path to directory with images
    :param test_file_path: file with test image paths and labels
    :param batch_size: test batch size
    :param device: 'cpu' or 'cuda'
    '''

    # load configurations from saved model, initialize and test the model
    checkpoint = torch.load(model_path)
    configs = checkpoint['configs']
    norm_size = configs['norm_size']
    output_size = configs['output_size']
    hidden_size = configs['hidden_size']
    n_layers = configs['n_layers']
    act_type = configs['act_type']

    # initialize the model by MLP()
    model = MLP(norm_size[0] * norm_size[1], output_size,
                hidden_size, n_layers, act_type)
    
    # load model parameters we saved in model_path
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    print('[Info] Load model from {}'.format(model_path))

    # enter the evaluation mode
    model.eval()

    # test loader
    testloader = dataLoader(im_dir, test_file_path, norm_size, batch_size)

    # run the test process
    n_correct = 0.
    n_ims = 0.
    logits = []
    all_labels = []

    with torch.no_grad():  # we do not need to compute gradients during test stage

        for ims, labels in testloader:
            ims, labels = ims.to(device), labels.type(torch.float).to(device)
            input = ims.view(ims.size(0), -1)
            out = model(input)
            predictions = out.argmax(1)
            n_correct += torch.sum(predictions == labels)
            n_ims += ims.size(0)
            logits.append(out)
            all_labels.append(labels)
        logits = torch.cat(logits, dim=0).detach().cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

        tsne = TSNE(n_components=2, init='pca')
        Y = tsne.fit_transform(logits)

        letters = list(string.ascii_letters[-26:])
        Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0))
        for i in range(len(all_labels)):
            if(all_labels[i]<26):
                c = plt.cm.rainbow(float(all_labels[i])/26)
                plt.text(Y[i, 0], Y[i, 1], s=letters[int(all_labels[i])], color=c)
        plt.show()
    print('[Info] Test accuracy = {:.1f}%'.format(100 * n_correct / n_ims))


# ==== Part 5: predict new images
def predict(model_path, im_path):
    '''
    Predict procedure. We predict one picture.
    ---------------
    :param model_path: path of the saved model
    :param im_path: path of an image
    '''

    # TODO 3: load configurations from saved model, initialize the model. 
    # Note: you can complete this section by referring to Part 4: test. 
    # load configurations from saved model, initialize and test the model
    # step 1: load configurations from saved model using torch.load(model_path)
    checkpoint = torch.load(model_path)
    # and get the configs dictionary, configs = checkpoint['configs'],
    configs = checkpoint['configs']
    # then get each config from configs, eg., norm_size = configs['norm_size']
    norm_size = configs['norm_size']
    output_size = configs['output_size']
    hidden_size = configs['hidden_size']
    n_layers = configs['n_layers']
    act_type = configs['act_type']
    # step 2: initialize the model by MLP()
    model = MLP(norm_size[0] * norm_size[1], output_size,
                hidden_size, n_layers, act_type)
    # step 3: load model parameters we saved in model_path
    # hint: similar to what we do in Part 4: test.
    model.load_state_dict(checkpoint['state_dict'])
    print('[Info] Load model from {}'.format(model_path))
    # End TODO 3
    # enter the evaluation mode
    model.eval()

    # image pre-processing, similar to what we do in ListDataset()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(norm_size),
        transforms.ToTensor()
    ])

    # image pre-processing, similar to what we do in ListDataset()
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = transform(im)
    im.sub_(0.5).div_(0.5)

    # input im into the model
    with torch.no_grad():
        input = im.view(1, -1)
        out = model(input)
        prediction = out.argmax(1)[0].item()

    # convert index of prediction to the corresponding character
    letters = string.ascii_letters[-26:]  # ABCD...XYZ
    prediction = letters[prediction]

    print('Prediction: {}'.format(prediction))


# ==== Part 6: draw the loss curve
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


if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 2021
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # set configurations
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, test or predict')
    parser.add_argument('--im_dir', type=str, default='data/character_classification/images',
                        help='path to directory with images')
    parser.add_argument('--train_file_path', type=str, default='data/character_classification/train.json',
                        help='file list of training image paths and labels')
    parser.add_argument('--val_file_path', type=str, default='data/character_classification/validation.json',
                        help='file list of validation image paths and labels')
    parser.add_argument('--test_file_path', type=str, default='data/character_classification/test.json',
                        help='file list of test image paths and labels')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')

    # configurations for training
    parser.add_argument('--hsize', type=str, default='32', help='hidden size for each hidden layer, splitted by comma')
    parser.add_argument('--layer', type=int, default=2, help='number of layers in the MLP')
    parser.add_argument('--act', type=str, default='relu',
                        help='type of activation function, can be none, sigmoid, tanh, or relu')
    parser.add_argument('--norm_size', type=str, default="32,32",
                        help='image normalization size, height and width, splitted by comma')
    parser.add_argument('--epoch', type=int, default=50, help='number of training epochs')
    parser.add_argument('--n_classes', type=int, default=26, help='number of classes')
    parser.add_argument('--valInterval', type=int, default=10, help='the frequency of validation')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--optim_type', type=str, default='sgd', help='type of optimizer, can be sgd, adagrad, rmsprop, adam, or adadelta')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum of the SGD optimizer, only used if optim_type is sgd')
    parser.add_argument('--weight_decay', type=float, default=0., help='the factor of L2 penalty on network weights')

    # configurations for test and prediction
    parser.add_argument('--model_path', type=str, default='saved_models/recognition.pth', help='path of a saved model')
    parser.add_argument('--im_path', type=str, default='data/character_classification/new_images/predict01.png',
                        help='path of an image to be recognized')

    opt = parser.parse_args()

    # -- run the code for training and validation
    if opt.mode == 'train':
        train_val(im_dir=opt.im_dir,
                  train_file_path=opt.train_file_path,
                  val_file_path=opt.val_file_path,
                  hidden_size=[int(n) for n in opt.hsize.split(',')],
                  n_layers=opt.layer,
                  act_type=opt.act,
                  norm_size=tuple(map(int, opt.norm_size.split(','))),
                  n_epochs=opt.epoch,
                  batch_size=opt.batchsize,
                  n_letters=opt.n_classes,
                  lr=opt.lr,
                  optim_type=opt.optim_type,
                  momentum=opt.momentum,
                  weight_decay=opt.weight_decay,
                  valInterval=opt.valInterval,
                  device=opt.device)

    # -- test the saved model
    elif opt.mode == 'test':
        test(model_path=opt.model_path,
             im_dir=opt.im_dir,
             test_file_path=opt.test_file_path,
             batch_size=opt.batchsize,
             device=opt.device)

    # -- predict a new image
    elif opt.mode == 'predict':
        predict(model_path=opt.model_path,
                im_path=opt.im_path)

    else:
        print('mode should be train, test, or predict')
        raise NotImplementedError
