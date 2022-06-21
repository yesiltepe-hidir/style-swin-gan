from sklearn.utils import shuffle
import torch
import torchvision
from torchvision import transforms

def get_data_loader(datasetname, root, batch_size):
  if datasetname == 'LSUN':
      dataset = torchvision.datasets.LSUN(
                    root = root,
                    classes = ['church_outdoor_train'],
                    transform = transforms.Compose([
                              transforms.Resize((128,128)),
                              # transforms.CenterCrop(64),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])
                    )
      dataloader = torch.utils.data.DataLoader(
                              dataset,
                              batch_size = batch_size,
                              num_workers = 2,
                              pin_memory = True,
                              shuffle=True
      )
  
  elif datasetname == 'CIFAR-10':
      dataset = torchvision.datasets.CIFAR10(
                    root = root,
                    train=True,
                    download=True,
                    transform = transforms.Compose([
                              transforms.Resize(32),
                              transforms.CenterCrop(32),
                              transforms.ToTensor()
                              ])
                    )
      dataloader = torch.utils.data.DataLoader(
                              dataset,
                              batch_size = batch_size,
                              num_workers = 2,
                              pin_memory = True
      )

  elif datasetname == 'Celeb-A':
    dataset = torchvision.datasets.CelebA(
                    root = root,
                    transform = transforms.Compose([
                              transforms.Resize((128,128)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                              ])
                    )
    dataloader = torch.utils.data.DataLoader(
                              dataset,
                              batch_size = batch_size,
                              num_workers = 2,
                              pin_memory = True,
                              shuffle=True
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
