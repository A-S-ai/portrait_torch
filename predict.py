import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tfs
from PIL import Image

from model import UNet
from utils import plot_img_and_mask


def predict_img(model, img, device, out_threshold=0.5):

    model.eval()
    img = img.unsqueeze(0)
    img = img.to(device=device)

    with torch.no_grad():
        output = model(img)
        print(output)
        print(torch.max(output))

        if net.n_classes > 1:
            pred = F.softmax(output, dim=1)
        else:
            pred = torch.sigmoid(output)

        pred = pred.squeeze(0)
        print(torch.max(pred))

        # trans = tfs.Compose([
        #         tfs.ToPILImage(),
        #         tfs.Resize(img.size[1]),
        #         tfs.ToTensor()
        # ])

        pred = pred.cpu()
        full_mask = pred.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='./result/MODEL-1.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()

    in_files = args.input
    out_files = get_output_filenames(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=1).to(device)

    logging.info("Loading model {}".format(args.model))
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn).convert('RGB')
        img = tfs.ToTensor()(img)

        mask = predict_img(model=net, img=img, out_threshold=args.mask_threshold, device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])
            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
