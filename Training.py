import sys

if len(sys.argv[1:]) != 0:
    d = int(sys.argv[1:][0])
else:
    d=1

# prerequisites
import torch
import os

import matplotlib.pyplot as plt
if d >=2:
    from mVAE_rec3rd import train, test, vae, optimizer, load_checkpoint
else:
    from mVAE import train, test, vae, optimizer, load_checkpoint
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import Dataset

checkpoint_folder_path = f'output_mnist_2drecurr{d}' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

if len(sys.argv[1:]) != 0:
    d = int(sys.argv[1:][0])
else:
    d=1
print(f'Device: {d}')

if torch.cuda.is_available():
    device = torch.device(f'cuda:{d}')
    torch.cuda.set_device(d)
    print('CUDA')
else:
    device = 'cpu'

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
load_checkpoint(f'{checkpoint_folder_path}/checkpoint_most_recent.pth', d)
print('checkpoint loaded')

bs=200

# trainging datasets, the return loaders flag is False so the datasets can be concated in the dataloader
#emnist_transforms = {'retina':True, 'colorize':True}
mnist_transforms = {'retina':True, 'colorize':True, 'scale':False, 'location_targets':{'left':[0,1,2,3,4],'right':[5,6,7,8,9]}}
mnist_test_transforms = {'retina':True, 'colorize':True, 'scale':False, 'location_targets':{'right':[0,1,2,3,4],'left':[5,6,7,8,9]}}
skip_transforms = {'skip':True, 'colorize':True}

#emnist_dataset = Dataset('emnist', emnist_transforms)
mnist_dataset = Dataset('mnist', mnist_transforms)

#emnist_test_dataset = Dataset('emnist', emnist_transforms, train= False)
mnist_test_dataset = Dataset('mnist', mnist_test_transforms, train= False)

#emnist_skip = Dataset('emnist', skip_transforms)
mnist_skip = Dataset('mnist', skip_transforms)

#concat datasets and init dataloaders
train_loader_noSkip = mnist_dataset.get_loader(bs)
sample_loader_noSkip = mnist_dataset.get_loader(25)
test_loader_noSkip = mnist_test_dataset.get_loader(bs)
mnist_skip = mnist_skip.get_loader(bs)

vae.to(device)

loss_dict = torch.load(f'mvae_loss_data_recurr{d}.pt') # {'retinal_train':[], 'retinal_test':[], 'cropped_train':[], 'cropped_test':[]} #  # #
seen_labels = {}
for epoch in range(327, 1001):
    loss_lst, seen_labels = train(epoch, train_loader_noSkip, None, mnist_skip, test_loader_noSkip, sample_loader_noSkip, True, seen_labels)
    
    # save error quantities
    loss_dict['retinal_train'] += [loss_lst[0]]
    loss_dict['retinal_test'] += [loss_lst[1]]
    loss_dict['cropped_train'] += [loss_lst[2]]
    loss_dict['cropped_test'] += [loss_lst[3]]
    torch.save(loss_dict, f'mvae_loss_data_recurr{d}.pt')

    torch.cuda.empty_cache()
    if epoch in [50,80,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                 'labels': seen_labels
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_{str(epoch)}.pth')
    else:
        checkpoint =  {
            'state_dict': vae.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'labels': seen_labels
                }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_most_recent.pth')
