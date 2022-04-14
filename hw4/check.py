# ========================================================
#             Media and Cognition
#             Homework 4 Support Vector Machine
#             check.py
#             Tsinghua University
#             (C) Copyright 2021
# ========================================================

from svm_hw import SVM_HINGE, Linear, Hinge
import torch
from torch.autograd import gradcheck


def run():
    model = SVM_HINGE(2, C=1.0).double()
    x = torch.randn(50, 2, requires_grad=False).double()
    W = torch.randn(1, 2, requires_grad=True).double()
    b = torch.zeros(1, requires_grad=True).double()
    test = gradcheck(Linear.apply, (x, W, b), eps=1e-6, atol=1e-4)
    if test:
        print('Linear successully tested!')
    output = torch.randn(50, 1, requires_grad=True).double()
    W = torch.randn(1, 2, requires_grad=True).double()
    labels = torch.ones(1, requires_grad=False).double()
    C = torch.tensor([[1.0]], requires_grad=False).double()
    test = gradcheck(Hinge.apply, (output, W, labels, C), eps=1e-6, atol=1e-5)
    if test:
        print('Hinge successfully tested！')
    x = torch.randn(50, 2, requires_grad=False).double()
    labels = torch.ones(50, requires_grad=False).double()
    try:
        output, loss = model(x, labels)
        assert model.W.requires_grad is True
        assert model.b.requires_grad is True
        print('SVM_HINGE successfully tested！')
    except:
        raise Exception('Failed testing SVM_HINGE!')


if __name__ == '__main__':
    run()