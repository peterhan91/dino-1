import os
import imageio
import pandas as pd
import numpy as np
from tqdm import tqdm

from dataset.utils import get_fundus


if __name__ == '__main__':
    df = pd.read_csv("./csv/split_.csv")
    df = df[df['fold'] == 'test']
    files = df['path'].tolist()[::-1]
    # means = []
    path = '../fundus_dataset/img_512/test'
    for file in tqdm(files):
        try:
            image = get_fundus(file, crop_size=512, mode='constant')
            image = (image - image.min()) / (image.max() - image.min())
            # means.append(image.mean((0, 1)))

            img = np.clip(np.rint(image * 255.0), 0.0, 255.0).astype(np.uint8)
            imageio.imwrite(os.path.join(path, os.path.basename(file)), img)
        except:
            print('Can not process ', file)

    # np.save('meanstd.npy', np.array(means))
