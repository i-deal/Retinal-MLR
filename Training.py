# prerequisites
import torch
import os
import matplotlib.pyplot as plt
from mVAE import train, test, vae, optimizer, load_checkpoint
from torch.utils.data import DataLoader, ConcatDataset
from dataset_builder import Dataset

checkpoint_folder_path = 'output_emnist_recurr' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
# load_checkpoint('CHECKPOINTNAME.pth')

bs=100

# trainging datasets, the return loaders flag is False so the datasets can be concated in the dataloader
emnist_transforms = {'retina':True, 'colorize':True}
mnist_transforms = {'retina':True, 'colorize':True, 'location_targets':{'left':[0,1,2,3,4],'right':[5,6,7,8,9]}}
skip_transforms = {'skip':True, 'colorize':True}

emnist_dataset = Dataset('emnist', emnist_transforms)
mnist_dataset = Dataset('mnist', mnist_transforms)

emnist_test_dataset = Dataset('emnist', emnist_transforms, train= False)
mnist_test_dataset = Dataset('mnist', mnist_transforms, train= False)

emnist_skip = Dataset('emnist', skip_transforms)
mnist_skip = Dataset('mnist', skip_transforms)

#concat datasets and init dataloaders
train_loader_noSkip = torch.utils.data.DataLoader(dataset=ConcatDataset([emnist_dataset, mnist_dataset, mnist_dataset]), batch_size=bs, shuffle=True,  drop_last= True)
test_loader_noSkip = torch.utils.data.DataLoader(dataset=ConcatDataset([emnist_test_dataset, mnist_test_dataset, mnist_test_dataset]), batch_size=bs, shuffle=True, drop_last=True)
emnist_skip = torch.utils.data.DataLoader(dataset=ConcatDataset([emnist_skip, mnist_skip, mnist_skip]), batch_size=bs, shuffle=True,  drop_last= True)
mnist_skip = mnist_skip.get_loader(bs)


loss_dict = {'retinal_train':[], 'retinal_test':[], 'cropped_train':[], 'cropped_test':[]}

for epoch in range(1, 301):
    loss_lst = train(epoch, train_loader_noSkip, emnist_skip, mnist_skip, test_loader_noSkip, True)
    
    # save error quantities
    loss_dict['retinal_train'] += [loss_lst[0]]
    loss_dict['retinal_test'] += [loss_lst[1]]
    loss_dict['cropped_train'] += [loss_lst[2]]
    loss_dict['cropped_test'] += [loss_lst[3]]
    torch.save(loss_dict, 'mvae_loss_data_recurr.pt')

    torch.cuda.empty_cache()
    if epoch in [50,80,100,150,200,250,300]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                 #'labels': seen_labels
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_{str(epoch)}.pth')