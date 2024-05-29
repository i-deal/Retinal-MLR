# prerequisites
import torch
import os
import matplotlib.pyplot as plt
from mVAE import train, test, vae, optimizer, load_checkpoint
from dataset_builder import Dataset

checkpoint_folder_path = 'output_red_green' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
# load_checkpoint('CHECKPOINTNAME.pth')

bs=100

# trainging datasets, the return loaders flag is False so the datasets can be concated in the dataloader
#emnist_dataset, emnist_skip, emnist_test_dataset = dataset_builder('emnist', bs, None, True, {'left':list(range(0,100)),'right':[]}, False)
#fmnist_dataset, fmnist_skip, fmnist_test_dataset = dataset_builder('fashion_mnist', bs, None, True, None, False)
transforms_train = {'colorize':True, 'color_targets':{0:[0,1,2,3,4],1:[5,6,7,8,9]}, 'retina':True}
mnist_dataset= Dataset('mnist', transforms_train)

transforms_test = {'colorize':True, 'color_targets':{1:[0,1,2,3,4],0:[5,6,7,8,9]}, 'retina':True}
mnist_test_dataset= Dataset('mnist', transforms_test, train=False)

print(mnist_dataset.all_possible_labels())

#concat datasets and init dataloaders
train_loader_noSkip = mnist_dataset.get_loader(100)
test_loader_noSkip = mnist_test_dataset.get_loader(100)

loss_dict = {'retinal_train':[], 'retinal_test':[], 'cropped_train':[], 'cropped_test':[]}

for epoch in range(1, 301):
    loss_lst = train(epoch, train_loader_noSkip,None, None, test_loader_noSkip, True)
    
    # save error quantities
    loss_dict['retinal_train'] += [loss_lst[0]]
    loss_dict['retinal_test'] += [loss_lst[1]]
    loss_dict['cropped_train'] += [loss_lst[2]]
    loss_dict['cropped_test'] += [loss_lst[3]]
    torch.save(loss_dict, 'mvae_loss_data_rg.pt')

    torch.cuda.empty_cache()
    if epoch in [50,80,100,150,200,250,300]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_{str(epoch)}.pth')