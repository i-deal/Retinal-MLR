# prerequisites
from dataset_builder import Dataset
#from dataset_builder_old import dataset_builder
from torchvision import utils
import time
import torch

transform = {'retina':True, 'colorize':True, 'scale':True, 'location_targets':{'left':[0,1,2,3], 'right':[4,5,6,7]}, 'color_targets':{4:[4,3,7], 1:[8,1], 2:[5,6]}, 
             'scale_targets': {0:[0,1,2,3,4],1:[5,6,7,8,9]}}
train_data = Dataset('mnist', transform)
data_ranges = train_data.all_possible_labels()
#print(data_ranges)
train_loader = train_data.get_loader(110)
data,labels = next(iter(train_loader))

#print(labels[0][0:100])
utils.save_image(data[0][0:10],'datasettest.png')
s = 0
for i in range(len(labels[0])):
    shape = labels[0][i].item()
    color = labels[1][i].item()
    retina = labels[2][i].item()

    x = data_ranges[shape] #[ [a,b], [c,d] ]

    if color not in list(range(x[0][0], x[0][len(x[0])-1]+1)):
        print(color, x[0])
    else:
        s += 0.5
    
    if retina not in list(range(x[1][0], x[1][len(x[1])-1]+1)):
        print(retina, x[1])
    else:
        s += 0.5

print(f'{int(s)}/{i+1}')

labels1 = [(1,2,4),(1,2,4),(3,3,2)]
labels2 = [(1,2,2),(3,3,2)]
print(set(labels1) | set(labels2)|set({}))