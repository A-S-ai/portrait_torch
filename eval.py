import torch
import torch.nn.functional as F
from tqdm import tqdm

from loss import dice_coeff


def evaluate(model, dataloader, num_classes, device, num_val):
    """ Evaluation without the densecrf with the dice coefficient """

    model.eval()
    total = 0

    with tqdm(total=num_val, desc='Validation round', leave=False) as t:

        for img, mask in dataloader:

            mask_type = torch.float32 if num_classes == 1 else torch.long
            img = img.to(device=device, dtype=torch.float32)
            mask = mask.to(device=device, dtype=mask_type)

            with torch.no_grad():
                pred = model(img)
            # for mask_, pred_ in zip(mask, pred):
            #     pred_ = (pred_ > 0.5).float()
            #     if model.n_classes > 1:
            #         tot += F.cross_entropy(pred_.unsqueeze(dim=0), mask_.unsqueeze(dim=0)).item()  # ??????
            #     else:
            #         tot += dice_coeff(pred, mask.squeeze(dim=1)).item()

            if num_classes > 1:
                total += F.cross_entropy(pred, mask).item()
            else:
                pred = torch.sigmoid(pred)
                pred = (pred > 0.5).float()
                total += dice_coeff(pred, mask).item()

            t.update(img.shape[0])

    return total / num_val
