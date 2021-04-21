import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm

from eval import eval_net
from model import UNet
from utils import time_str, show_losses
from dataset import BasicDataset


dir_img = 'data/supervisely/imgs/'
dir_mask = 'data/supervisely/masks/'


def save(epoch, losses):
    t_str = time_str()
    torch.save(model.state_dict(), './result/model_epoch{}'.format(epoch) + t_str + '.pth')
    show_losses('./log/loss_epoch{}'.format(epoch) + t_str + '.png', losses=losses)
    logging.info('Checkpoint saved !')


def train(model, device, epochs, batch_size, lr, val_percent=0., num_workers=2):

    dataset = BasicDataset(dir_img, dir_mask)
    num_val = int(len(dataset) * val_percent)
    num_train = len(dataset) - num_val

    train_date, val_data = random_split(dataset, [num_train, num_val])
    train_loader = DataLoader(train_date, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size, shuffle=False, num_workers=num_workers)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training item:   {num_train}
        Validation item: {num_val}
        Device:          {device.type}''')

    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)

    losses = []
    global_step = 0

    for epoch in range(epochs):
        model.train()

        with tqdm(total=num_train, unit='img') as t:
            t.set_description('epoch: {}/{}'.format(epoch + 1, epochs))

            for img, mask in train_loader:
                mask_type = torch.float32 if model.n_classes == 1 else torch.long

                # import cv2
                # print(mask.shape)
                # cv2.imshow('figure1', img[0].permute(1, 2, 0).numpy())
                # cv2.waitKey()
                # cv2.imshow('figure2', mask[0].permute(1, 2, 0).numpy())
                # cv2.waitKey()

                # mask = mask.to(device=device, dtype=mask_type)
                img = img.to(device=device)
                mask = mask.to(device=device)
                # print(img.dtype)
                # print(mask.dtype)
                # update
                pred = model(img)
                # print(pred.shape)
                optimizer.zero_grad()
                loss = criterion(pred, mask)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

                writer.add_scalar('Loss/train', loss.item(), global_step)

                t.set_postfix(loss='{:.6f}'.format(loss))
                t.update(img.shape[0])

                global_step += 1

                if global_step % (len(dataset) // (10 * batch_size)) == 0:
                    val_score = eval_net(model, val_loader, device, num_val)
                    scheduler.step(val_score)
                    if model.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', img, global_step)
                    if model.n_classes == 1:
                        writer.add_images('masks/true', mask, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(pred) > 0.5, global_step)

        save(epoch+1, losses)

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-e', '--epochs', metavar='E', type=int, default=20, help='Number of epochs', dest='epochs')
    parser.add_argument(
        '-b', '--batch-size', metavar='B', type=int, nargs='?', default=1, help='Batch size', dest='batchsize')
    parser.add_argument(
        '-lr', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001, help='Learning rate', dest='lr')
    parser.add_argument(
        '-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument(
        '-v', '--validation', dest='val', type=float, default=0.1, help='Percent of the data that is used as validation')
    return parser.parse_args()


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_properties(device))
    print("id：{}".format(device))
    print("name:{}".format(torch.cuda.get_device_name(0)))
    logging.info(f'Using device {device}')

    # n_channels=3 for RGB
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    model = UNet(n_channels=3, n_classes=1).to(device)
    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Dilated conv"} upscaling')

    train(model, device, args.epochs, args.batchsize, args.lr, val_percent=args.val)
