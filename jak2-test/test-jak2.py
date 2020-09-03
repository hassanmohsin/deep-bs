import numpy as np
import pandas as pd
from tqdm import tqdm

from data.data_loader import CreateDataLoader
from models.models import create_model
from options.test_options import TestOptions

opt = TestOptions().parse()
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)


def correlation(Measure, Fit):
    """Calculates the correlation coefficient R^2 between the two sets
       of Y data provided. Logically, in order for the result to have a sense
       you want both Y arrays to have been created from the same X array."""

    Mean = np.mean(Measure)
    s1 = 0
    s2 = 0
    Size = np.size(Measure)  # identical to np.size(Fit)

    for i in range(0, Size):
        s1 += (Measure[i] - Fit[i]) ** 2
        s2 += (Measure[i] - Mean) ** 2
    Rsquare = 1 - s1 / s2
    return Rsquare


# test
preds_pose = np.zeros(len(dataset))
trues_pose = np.zeros(len(dataset))
preds_affinity = np.zeros(len(dataset))
trues_affinity = np.zeros(len(dataset))
with tqdm(total=int(len(dataset) / opt.batch_size) + 1) as pbar:
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        offset = i * opt.batch_size
        preds_pose[offset:offset + opt.batch_size] = model.preds_pose.cpu().detach().numpy().flatten()
        trues_pose[offset:offset + opt.batch_size] = data['pose'].flatten()
        preds_affinity[offset:offset + opt.batch_size] = model.preds_affinity.cpu().detach().numpy().flatten()
        trues_affinity[offset:offset + opt.batch_size] = data['affinity'].flatten()
        pbar.update()

# from sklearn.metrics import r2_score
# print("corr coef:", np.corrcoef(preds, trues)[0,1])
# print("R2:", r2_score(trues, preds))

pd.DataFrame(np.vstack((trues_pose, preds_pose)).T, columns=('true', 'pred')).to_csv('output_pose.csv', index=False)
pd.DataFrame(np.vstack((trues_affinity, preds_affinity)).T, columns=('true', 'pred')).to_csv('output_affinity.csv',
                                                                                             index=False)
