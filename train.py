from monai.networks.nets import UNet, UNETR, SegResNetDS, DynUNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss
import os
import torch
from preprocess import prepare
from utilities import train


# Define directories
data_dir = r'C:\Users\uzivatel\OneDrive - Ostravská univerzita\Plocha\Pytorch\datasets\Atlas_prepared_extracted'
model_dir = r'C:\Users\uzivatel\OneDrive - Ostravská univerzita\Plocha\Pytorch\datasets\Atlas_prepared_extracted\results' 
#pretrained_path = 
data_in = prepare(data_dir)

# Define device
device = torch.device("cuda:0")

spatial_dims = 3
in_channels = 1  # Assuming single-channel MRI scans
out_channels = 2  # Number of segmentation classes

'''
# Kernel sizes and strides are chosen to maintain compatibility with the input size
kernel_size = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
strides = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
upsample_kernel_size = [(2, 2, 2), (2, 2, 2), (2, 2, 2)]

# Filters for each block
filters = [32, 64, 128, 256]

model = DynUNet(
    spatial_dims=spatial_dims,
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    strides=strides,
    upsample_kernel_size=upsample_kernel_size,
    filters=filters,
    dropout=None,  
    norm_name=('INSTANCE', {'affine': True}),
    act_name=('leakyrelu', {'inplace': True, 'negative_slope': 0.01}),
    deep_supervision=False,
    deep_supr_num=1,
    res_block=False,
    trans_bias=False
).to(device)




model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.2
).to(device)


# Define model
model = UNETR(in_channels=1, 
              out_channels=2, 
              img_size=(128,128,128), 
              feature_size=32,
              norm_name='batch').to(device)
'''
              
model = SegResNetDS(spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels
                ).to(device)

# Use DataParallel to distribute the model across GPUs

#model = torch.nn.DataParallel(model).cuda()

# Define loss function and optimizer
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

# Training function
if __name__ == '__main__':
    # Clear cache before training
    torch.cuda.empty_cache() 
    # Train the model
    train(model, data_in, loss_function, optimizer, 200, model_dir, device=device)
