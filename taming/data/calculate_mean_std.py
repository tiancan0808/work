import glob, os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

batch_size = 20

class dagm(Dataset):
    def __init__(self, src, size=256,img_size=224, **kwargs):
        super().__init__()
        self.src = src
        self.size=size
        self.transform = transforms.Compose([transforms.Resize(size),transforms.CenterCrop(img_size),transforms.ToTensor()])
        
        self.paths = sorted(glob.glob(os.path.join(src,'train', f'*.PNG')))
                            
    def __getitem__(self, idx):
        img = self.transform(Image.open(self.paths[idx]).convert('L'))
        return img

    def __len__(self):
        return len(self.paths)

data = dagm('../../../DAGM/class7')

dataloader = DataLoader(
    data,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True,
    pin_memory=False
)
def get_mean_std_value(loader):
    '''
    求数据集的均值和标准差
    :param loader:
    :return:
    '''
    data_sum,data_squared_sum,num_batches = 0,0,0

    for data in loader:
        # data: [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的均值和，dim=1为通道数量，不用参与计算
        data_sum += torch.mean(data,dim=[0,2,3])    # [batch_size,channels,height,width]
        # 计算dim=0,2,3维度的平方均值和，dim=1为通道数量，不用参与计算
        data_squared_sum += torch.mean(data**2,dim=[0,2,3])  # [batch_size,channels,height,width]
        # 统计batch的数量
        num_batches += 1
    # 计算均值
    mean = data_sum/num_batches
    # 计算标准差
    std = (data_squared_sum/num_batches - mean**2)**0.5
    return mean,std

np.set_printoptions(precision=8,suppress=False)
mean,std = get_mean_std_value(dataloader)
print('mean = {},std = {}'.format(mean.numpy(),std.numpy()))

## class1 mean = 0.27475 std = 0.05502
## class2 mean = 0.3957 std = 0.1276