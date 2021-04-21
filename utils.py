import matplotlib.pyplot as plt
from datetime import datetime


def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def show_losses(save_path, **kwargs):
    plt.figure(figsize=(10, 5))
    plt.title("Losses During Training")
    for key, value, in kwargs.items():
        plt.plot(value, label=key)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    print('saving loss at: {}'.format(save_path))


def time_str():
    return datetime.now().strftime('_%m%d%H%M')
