from __future__ import print_function, division

import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

from grasp_network import *
from grasp_data import *


def train(batch_size, model, device, train_loader, optimizer, criterion, method, epoch, writer=None):
    model.train()
    correct = 0
    counter = 0
    losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        x = [d.to(device) for d in data]
        target = target.to(device)

        optimizer.zero_grad()

        output = model(x)

        if method in ('pintogupta'):
            target = target.float()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if len(output.shape) == 1:
            pred = np.array([int(i > 0) for i in output.float()])
        else:
            pred = np.array([int(i < j) for i, j in output.float()])

        correct += np.sum(pred == target.cpu().numpy())
        counter += len(target)

        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx *
                batch_size, len(train_loader.dataset), 100. *
                batch_idx / len(train_loader),
                loss.item()))
            losses.append(loss.item())
    if writer:
        writer.add_scalar('data/training-loss', np.mean(losses), epoch)
        writer.add_scalar('data/training_accuracy', correct / counter, epoch)


def val(model, device, test_loader, epoch, prcurve_path, writer=None):
    model.eval()
    correct = 0
    scores = []
    targets = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            x = [d.to(device) for d in data]
            target = target.to(device)
            output = model(x)
            if len(output.shape) == 1:
                pred = []
                for out in output.float():
                    pred.append(int(out) > 0)
                    scores.append(out)
                predictions.append(pred)
            else:
                pred = np.array([int(i < j) for i, j in output.float()])
                predictions.append(pred)
            correct += np.sum(pred == target.cpu().numpy())
            targets.append(target.cpu().numpy())

    print('\nVal accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),
                                                     100. * correct / len(test_loader.dataset)))
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    if writer:
        writer.add_scalar('test-accuracy', correct /
                          len(test_loader.dataset), epoch)
    return correct / len(test_loader.dataset)


def get_probability(s0, s1):
    p0, p1 = np.exp(s0), np.exp(s1)
    return p1 / (p0 + p1)


def main(batch_size, epochs, lr, ignore_grasp, crop_radius, method,
         resultpath, start, weight_decay, datapath):

    assert method in ('fullimage', 'pintogupta')

    use_cuda = torch.cuda.is_available()
    print("Cuda available: " + str(use_cuda))

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    def crossval_split(folds, dsize, indices=[]):
        chunk_size = int(dsize / folds)

        if len(indices) == 0:
            indices = np.arange(dsize)

        np.random.shuffle(indices)

        def indices_to_mask(indices):
            mask_list = np.zeros((dsize))
            for i in indices:
                mask_list[i] = 1
            return np.array(mask_list)

        return [indices_to_mask(indices[chunk_size * i: chunk_size * (i + 1)])
                for i in range(folds)] + [1 for _ in range(dsize % folds)]

    def indices_to_mask(indices):
        mask_list = np.zeros((dataset_size))
        for i in indices:
            mask_list[i] = 1
        return np.array(mask_list)

    dataset_size = np.load(datapath + 'successes.npy').shape[0]
    samples = np.arange(dataset_size)
    np.random.shuffle(samples)
    train_size = int(dataset_size * .9)
    val_mask = indices_to_mask(samples[train_size:])
    train_mask = indices_to_mask(samples[:train_size])

    val_accs = []

    if method == 'fullimage':
        train_data = StandardData(
            path=datapath, valid_mask=train_mask, train=True)
        val_data = StandardData(
            path=datapath, valid_mask=val_mask, train=False)
    elif method == 'pintogupta':
        train_data = CroppedData(
            path=datapath, valid_mask=train_mask, crop_radius=crop_radius, train=True)
        val_data = CroppedData(
            path=datapath, valid_mask=val_mask, crop_radius=crop_radius, train=False)

    # weighting samples to correct imbalance with successes/failures in the
    # dataset
    hits = np.sum(train_data.successes)
    train_weights = [1 / (len(train_data) - hits) if elem ==
                     0 else 1 / hits for elem in train_data.successes]
    train_sampler = WeightedRandomSampler(
        train_weights, len(train_data), replacement=True)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, **kwargs)

    hits = np.sum(val_data.successes)
    val_weights = [1 / (len(val_data) - hits) if elem ==
                   0 else 1 / hits for elem in val_data.successes]
    val_sampler = WeightedRandomSampler(
        val_weights, len(val_data), replacement=True)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                            sampler=val_sampler, **kwargs)

    current_lr = lr
    current_lr /= 10**(int((start - 1) / 50))

    if method == 'fullimage':
        model = FullImageNet(ignore_grasp=ignore_grasp)
    elif method == 'pintogupta':
        model = PintoGuptaNet(binned_output=True)

    model.to(device)

    if use_cuda:
        model.cuda()
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")

    model = nn.DataParallel(model)

    if method in ('pintogupta'):
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    writer = SummaryWriter(log_dir=resultpath)

    if start > 0:
        model.load_state_dict(torch.load(
            resultpath + '/model-epoch-%d.th' % (start - 1)))
        optimizer = optim.Adam(model.parameters(), lr=current_lr)
        optimizer.load_state_dict(torch.load(
            resultpath + '/optimizer-epoch-%d.th' % (start - 1)))
    else:
        optimizer = optim.Adam(model.parameters(), lr=current_lr)

    for epoch in range(start, epochs + 1):
        if epoch % 50 == 0:
            current_lr /= 10
            optimizer = optim.Adam(model.parameters(), lr=current_lr,
                                   weight_decay=weight_decay)

        train(batch_size, model, device, train_loader,
              optimizer, criterion, args.method, epoch, writer)
        val_acc = val(model, device, val_loader, epoch,
                      resultpath + '/pr_curves', writer)
        torch.save(optimizer.state_dict(), resultpath +
                   'optimizer-epoch-%d.th' % epoch)
        torch.save(model.state_dict(), resultpath +
                   'model-epoch-%d.th' % epoch)

    writer.close()
    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grasp evaluation')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--ignore_grasp', type=int, default=0)
    parser.add_argument('--crop_radius', type=int, default=48,
                        help='Radius to crop image around for Pinto')
    parser.add_argument('--method', type=str, default='fullimage')
    parser.add_argument('--resultpath', type=str, default='models/')
    parser.add_argument('--start', type=int, default=0,
                        help='Epoch to start training from. If start != 0, will load a saved checkpoint in your resultpath')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--datapath', type=str,
                        default='')
    args = parser.parse_args()

    main(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
         ignore_grasp=args.ignore_grasp, crop_radius=args.crop_radius, method=args.method,
         resultpath=args.resultpath, start=args.start, weight_decay=args.weight_decay, datapath=args.datapath)
