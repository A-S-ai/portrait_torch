import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch import optim
from tqdm import tqdm
from os.path import join

from eval import evaluate
from model import UNet
from utils import time_str, show_losses
from dataset import SuperviselyDataset
from config import Config


def save_model_and_loss(model, moder_root, losses, loss_root):
    t_str = time_str()
    torch.save(model.state_dict(), join(moder_root, 'model' + t_str + '.pth'))
    show_losses(join(loss_root, 'loss' + t_str + '.png'), losses=losses)
    logging.info('Checkpoint & Losses saved !')


def train(n_channels, n_classes, bilinear, epochs, batch_size, lr, val_rate, num_workers, pin_memory, roots, threshold):

    data_root, model_root, log_root = roots

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_properties(device))
    print("id：{}".format(device))
    print("name:{}".format(torch.cuda.get_device_name(0)))
    logging.info(f'Using device {device}')

    model = UNet(n_channels, n_classes, bilinear).to(device)
    logging.info(f'Network:\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if bilinear else "Dilated conv"} upscaling')

    dataset = SuperviselyDataset(data_root)
    num_val = int(len(dataset) * val_rate)
    num_train = len(dataset) - num_val

    train_date, val_data = random_split(dataset, [num_train, num_val])
    train_loader = DataLoader(train_date, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')

    logging.info(f'''Starting training:
        Epochs:                     {epochs}
        Batch size:                 {batch_size}
        Learning rate (original):   {lr}
        Training item:              {num_train}
        Validation item:            {num_val}
        Device:                     {device.type}''')

    # criterion = nn.CrossEntropyLoss() if n_classes > 1 else nn.BCEWithLogitsLoss()
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if n_classes > 1 else 'max', patience=2)

    losses = []
    global_step = 0

    for epoch in range(epochs):
        model.train()

        with tqdm(total=num_train) as t:
            t.set_description('epoch: {}/{}'.format(epoch + 1, epochs))

            for img, mask in train_loader:
                # import cv2
                # print(mask.shape)
                # cv2.imshow('figure1', img[0].permute(1, 2, 0).numpy())
                # cv2.waitKey()
                # cv2.imshow('figure2', mask[0].permute(1, 2, 0).numpy())
                # cv2.waitKey()

                mask_type = torch.float32 if n_classes == 1 else torch.long
                mask = mask.to(device=device, dtype=mask_type)
                img = img.to(device=device, dtype=torch.float32)

                # update
                pred = model(img)
                loss = criterion(pred, mask)

                optimizer.zero_grad()
                nn.utils.clip_grad_value_(model.parameters(), 0.1)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                # writer.add_scalar('Loss/train', loss.item(), global_step)

                t.set_postfix(loss='{:.6f}'.format(loss), lr='%.8f' % optimizer.param_groups[0]['lr'])
                t.update(img.shape[0])

                # value
                # global_step += 1
                # if global_step % (len(dataset) // (10 * batch_size)) == 0:

            # score = evaluate(model, val_loader, n_classes, device, num_val)
            # model.train()
            # logging.info('Validation Dice Coeff: {}'.format(score))
            # scheduler.step(score)

                    # if n_classes > 1:
                    #     logging.info('Validation cross entropy: {}'.format(score))
                    #     writer.add_scalar('Loss/test', score, global_step)
                    # else:
                    #     logging.info('Validation Dice Coeff: {}'.format(score))
                    #     writer.add_scalar('Dice/test', score, global_step)
                    #
                    # writer.add_images('images', img, global_step)
                    # if n_classes == 1:
                    #     writer.add_images('masks/true', mask, global_step)
                    #     writer.add_images('masks/pred', torch.sigmoid(pred) > threshold, global_step)

        save_model_and_loss(model, model_root, losses, log_root)

    writer.close()


parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
parser.add_argument('-s', '--server', action='store_true', help='Whether use server training')
# parser.add_argument('-d', '--dataset', type=str, default='', help='dataset name')
args = parser.parse_args()


if __name__ == '__main__':
    cf = Config(args.server)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    train(cf.num_channels, cf.num_classes, cf.bilinear, cf.epoch, cf.batch_size,
          cf.lr, cf.val_rate, cf.num_workers, cf.pin_m, cf.roots(), cf.threshold)
