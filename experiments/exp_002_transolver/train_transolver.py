from comet_ml import Experiment
import numpy as np
import pandas as pd
import xarray as xr
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
from tools.parse_data import parse_data
import torch
from TransolverPDE.model.Transolver_Structured_Mesh_2D import Model
from TransolverPDE.utils.testloss import TestLoss


if torch.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Using device:", device)

POLLUTION_DIR = pathlib.Path("data/pollution")
dataset_paths = [str(path) for path in POLLUTION_DIR.glob("*")]

xr_tmp = xr.open_dataset(dataset_paths[1])


def transform_func(item: np.ndarray):
    if item.shape == (18, 251, 201):
        return item
    # elif item.shape == (18, 3, 300, 300):
    #     return [item[:, 0, :251, :201], item[:, 1, :251, :201], item[:, 2, :251, :201]]
    else:
        return []


data, source_coordinates = parse_data(dataset_paths, transform_func=transform_func)
data = np.transpose(data, (0, 2, 3, 1))

T_in = 18
T = 8
# h = 101
r = 2
ntrain = 400
ntest = 90
step = 1
# todo: отнормировать масштаб в интервал от 0 до 1

train_a = data[:ntrain, :, :, :]
train_a_coords = source_coordinates[:ntrain]
train_a = train_a.reshape(train_a.shape[0], -1, train_a.shape[-1])
train_a = torch.from_numpy(train_a)
train_a_coords = torch.from_numpy(train_a_coords)

# train_u = data[:ntrain, :, :, :]
# train_u = train_u.reshape(train_u.shape[0], -1, train_u.shape[-1])
# train_u = torch.from_numpy(train_u)


test_a = data[-ntest:, :, :, :]
test_a_coords = source_coordinates[-ntest:]
test_a = test_a.reshape(test_a.shape[0], -1, test_a.shape[-1])
test_a = torch.from_numpy(test_a)
test_a_coords = torch.from_numpy(test_a_coords)

# test_u = data[-ntest:, :, :, :]
# test_u = test_u.reshape(test_u.shape[0], -1, test_u.shape[-1])
# test_u = torch.from_numpy(test_u)


x = np.linspace(0, 1, 251)
y = np.linspace(0, 1, 201)
x, y = np.meshgrid(x, y)
pos = np.c_[x.ravel(), y.ravel()]
pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
pos_train = pos.repeat(ntrain, 1, 1)
pos_test = pos.repeat(ntest, 1, 1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, train_a, train_a_coords),
                                           batch_size=8,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, test_a, test_a_coords),
                                          batch_size=8,
                                          shuffle=True)
print(f"Train samples: {len(train_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")

tmodel = Model(
    space_dim=2,
    n_layers=3,
    n_hidden=64,
    n_head=4,
    Time_Input=False,
    mlp_ratio=1,
    fun_dim=17,
    out_dim=1,
    slice_num=64,
    ref=8,
    unified_pos=1,
    H=201,
    W=251
).to(device)

optimizer = torch.optim.AdamW(tmodel.parameters())
myloss = TestLoss(size_average=False)

experiment = Experiment()
experiment.set_name("transolver-train")

for epoch in range(100):
    print(f"Epoch {epoch+1}")
    tmodel.train()
    L = 0.0
    for xb, yb, coords in tqdm(train_loader):
        xb, yb, coords = xb.to(device, torch.float32), yb.to(device, torch.float32), coords.to(device, torch.float32)
        pred = tmodel(xb, yb[..., 1:])
        loss = myloss(pred, yb[..., 0].reshape(yb.shape[0], yb.shape[1], 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        L += loss.item()

    L /= len(train_loader)

    tmodel.eval()
    test_L = 0.0
    with torch.no_grad():
        for xb, yb, coords in test_loader:
            xb, yb, coords = xb.to(device, torch.float32), yb.to(device, torch.float32), coords.to(device, torch.float32)
            pred = tmodel(xb, yb[..., 1:])
            loss = myloss(pred, yb[..., 0].reshape(yb.shape[0], yb.shape[1], 1))
            test_L += loss.item()

    test_L /= len(test_loader)

    print(f"Epoch {epoch:03d}  loss {L:.6f}  test_loss {test_L:.6f}")
    experiment.log_metrics({"train_loss": L, "test_loss": test_L}, step=epoch)

experiment.end()

tmodel.eval()

for x, y, coords in test_loader:
    # print(x.shape, y.shape)
    x, y, coords = x.to(device, torch.float32), y.to(device, torch.float32), coords.to(device, torch.float32)
    pred = tmodel(x, y[..., 1:])

    fig, ax = plt.subplots(ncols=2, figsize=(13, 6))

    # plt.figure()
    ax[0].imshow(pred[0, ...].reshape((251, 201)).detach().cpu().numpy())
    ax[0].set_title("Prediction")
    # plt.show()

    # plt.figure()
    ax[1].imshow(y[0, ..., 0].reshape((251, 201)).detach().cpu().numpy())
    ax[1].set_title("True")
    plt.show()


TMODEL_PATH = "transolver_weights_17_to_1_slava.pth"
torch.save(tmodel.state_dict(), TMODEL_PATH)
print(f"Saved weights to {TMODEL_PATH}")
