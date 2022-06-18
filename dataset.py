'''
Get LSUN Church dataset via following commands:

  >> git clone https://github.com/fyu/lsun.git
  >> python3 lsun/download.py -c church_outdoor
  >> unzip church_outdoor_train_lmdb.zip
'''

import torch
import torchvision
from torchvision import transforms

def get_data_loader(datasetname, root, batch_size):
  if datasetname == 'LSUN':
      dataset = torchvision.datasets.LSUN(
                    root = root,
                    classes = ['church_outdoor_train'],
                    transform = transforms.Compose([
                              transforms.Resize(256),
                              transforms.CenterCrop(256),
                              transforms.ToTensor()
                              ])
                    )
      dataloader = torch.utils.data.DataLoader(
                              dataset,
                              batch_size = batch_size,
                              num_workers = 2,
                              pin_memory = True
      )

  else:
    raise ValueError(f'No dataset named {datasetname}!')
  
  return dataloader

if '__name__' == '__main__':
  # Dataset
  datasetname = 'LSUN'
  # Data Root
  root = './'
  # Parameters
  batch_size = 256
  # Get Dataloader
  loader = get_data_loader(datasetname, root, batch_size)