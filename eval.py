import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff


def eval_net(model, loader, device, n_val):
    """ Evaluation without the densecrf with the dice coefficient """

    model.eval()
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as t:

        for img, mask in loader:
            # mask_type = torch.float32 if net.n_classes == 1 else torch.long

            img = img.to(device=device)
            mask = mask.to(device=device)
            # mask = mask.to(device=device, dtype=mask_type)

            pred = model(img)

            # for mask_, pred_ in zip(mask, pred):
            #     pred_ = (pred_ > 0.5).float()
            #     if model.n_classes > 1:
            #         tot += F.cross_entropy(pred_.unsqueeze(dim=0), mask_.unsqueeze(dim=0)).item()  # ??????
            #     else:
            #         tot += dice_coeff(pred, mask.squeeze(dim=1)).item()

            if model.n_classes > 1:
                tot += F.cross_entropy(pred, mask).item()
            else:
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, mask).item()

            t.update(img.shape[0])

    model.train()
    return tot / n_val
