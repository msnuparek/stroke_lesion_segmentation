from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from preprocess import prepare
from utilities import train


data_dir = 'C:/Users/uzivatel/OneDrive - Ostravská univerzita/Plocha/Pytorch/datasets/Atlas_prepared'
model_dir = 'C:/Users/uzivatel/OneDrive - Ostravská univerzita/Plocha/Pytorch/datasets/Atlas_prepared/results' 
data_in = prepare(data_dir)

device = torch.device("cpu")
model = UNet(
    space_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


#loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 1, model_dir)