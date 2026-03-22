import numpy as np
def dataloaderNumpy(loader):
    imgs, labels = [], []
    for x_batch, y_batch in loader:
        imgs.append(x_batch.numpy())
        labels.append(y_batch.numpy())
    imgs = np.vstack(imgs)
    labels = np.hstack(labels)
    return imgs, labels