import random
import torch


def train(model=None, dataloader=None, criterion=None, optimizer=None, device='cpu'):
    running_loss = 0.0
    num_iter = 0
    max_iter = len(dataloader)
    model = model.train()
    for i, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        labels = [(x.to(device), y.to(device)) for x, y in labels]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_iter += 1
        if (i+1) % 100 == 0:
            print('Iter [{}/{}], Loss = {}'.format(i + 1, max_iter, running_loss / num_iter))
    return running_loss / num_iter


def val(model=None, dataloader=None, criterion=None, device='cpu'):
    running_loss = 0.0
    num_iter = 0
    model = model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            labels = [(x.to(device), y.to(device)) for x, y in labels]
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            num_iter += 1
    return running_loss / num_iter


def getDistortions(num_fragments=10, random_values=False):
    interval = 0.9 / (num_fragments - 1)
    distortions = [interval * i for i in range(num_fragments)]
    if random_values:
        # BUG FIX: only slot 0 (base value 0.0) can land below zero here,
        # since its sampling range is [0 - interval, 0]. Every other slot's
        # range stays >= 0. A real fisheye photo in this codebase's
        # convention only ever has kdist in [0, 0.9] (see the base grid
        # above) — a stray negative sample represents a distortion
        # direction (pincushion-like) that never occurs in real target
        # data, so it's wasted training signal on an out-of-domain regime.
        # Clamp to keep every sampled value within the physically
        # realistic range.
        distortions = [max(0.0, val - random.random() * interval) for val in distortions]
    return distortions
