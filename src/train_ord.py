import os
import numpy as np
import torch
torch.cuda.empty_cache()
from torch.autograd import Variable
# from models.attentionnet import Attn_Net
from custom_loss import OrdinalCrossEntropyLoss
from models.cnns import Custom_ResNet34
from torchvision.models.vgg import model_urls
import torchvision.models as models
from models.mobilenet import MobileNet
from models.mobilevit import MobileViT
from dataloader import MyJP2Dataset, NFDataset, FLDataset
from evaluation import sklearn_Compatible_preds_and_targets, accuracy_score
from prettytable import PrettyTable
# All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.nn as nn 
import torch.nn.functional as F

# For all Optimization algorithms, SGD, Adam, etc.
import torch.optim as optim

# Loading and Performing transformations on dataset
import torchvision
import torchvision.transforms as transforms 
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.sampler import Sampler, WeightedRandomSampler

#Labels in CSV and Inputs in Fits in a folder
import pandas as pd
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt

#For Confusion Matrix
from sklearn.metrics import confusion_matrix

#Warnings
import warnings
warnings.simplefilter("ignore", Warning)

#Time Computation
import timeit
import argparse
import copy
import yaml
import wandb

os.environ['WANDB_DISABLE_CODE'] = 'false'

wandb.init(project='tune_ordinal_reverse', save_code=True)
config = wandb.config

def str_to_bool(value):
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Invalid boolean value: {}'.format(value))

#Experiment CLI Options
parser = argparse.ArgumentParser(description="fullDiskModeltrainer")
parser.add_argument("--loss", type=str, default='ord_ce', help="Loss Selection")
parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
parser.add_argument("--saving_dir", type=str, default='ORDINAL', help="Enter model name for saving directory for corresponding model")
parser.add_argument("--datatype", type=str, default='augment', help="undersampled or augmented undersampled")
parser.add_argument("--models", type=str, default='resnet', help="enter resent, mobilenet, and mobilevit resp")

#Hyperparameters for Wandb
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=config.learning_rate, help="learning rate")
parser.add_argument("--weight_decay", type=float, default=config.weight_decay, help="regularization parameter")
parser.add_argument("--alpha", type=float, default=config.alpha, help="scaling parameter")
# parser.add_argument("--gamma", type=int, default=config.gamma, help="smoothing parameter")
# parser.add_argument("--max_lr", type=float, default=0.0001, help="MAX learning rate")

opt = parser.parse_args()

saving_directory = "/scratch/users/cpandey1/DSAA_EXPTS"
os.makedirs(saving_directory, exist_ok=True)

# job_id = os.getenv('SLURM_JOB_ID')
saving_directory = os.path.join(saving_directory, str(opt.saving_dir))
os.makedirs(saving_directory, exist_ok=True)

model_dir = os.path.join(saving_directory, "trained_models/")
os.makedirs(model_dir, exist_ok=True)

results_dir = os.path.join(saving_directory, "results/")
os.makedirs(results_dir, exist_ok=True)


# Random Seeds for Reproducibility
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(4)


# CUDA for PyTorch -- GPU SETUP
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print(torch.cuda.device_count())
# device= torch.device('cpu')
torch.backends.cudnn.benchmark = True
# print(device)

def get_augmented_dataset(csv_file, root_dir, transform):
    return FLDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)


def dataloading():

    #Labels Locations

    csv_file_train = f'/scratch/users/cpandey1/ar_data_labels/final_ecml/final/train_new_rev.csv'
    csv_file_val = f'/scratch/users/cpandey1/ar_data_labels/final_ecml/final/val_new_rev.csv'
    csv_file_test = f'/scratch/users/cpandey1/ar_data_labels/final_ecml/final/test_new_rev.csv'

    #Define Transformations
    transformations = transforms.Compose([
        transforms.ToTensor()
    ])

    
    #Specifying original images directories
    original_image_dir = '/scratch/users/cpandey1/stride_based_hourly_all/'

    if opt.datatype=='augment':

        # Parent directory Augmentations
        parent_dir = '/scratch/users/cpandey1/fl_augmentations_stride/'


        augmentation_dirs = ['adding_noise', 'gaussian_smoothing', 'horizontal_flip', 'vertical_flip', 'polarity_change']
        #     ]

        #Loading Dataset -- Trimonthly partitioned with defined augmentation
        ori_nf = NFDataset(csv_file = csv_file_train, 
                                    root_dir = original_image_dir,
                                    transform = transformations)

        ori_fl = FLDataset(csv_file = csv_file_train, 
                                    root_dir = original_image_dir,
                                    transform = transformations)

        augmented_datasets = []
        for aug_dir in augmentation_dirs:
            aug_root_dir = os.path.join(parent_dir, aug_dir)
            augmented_dataset = get_augmented_dataset(csv_file=csv_file_train, root_dir=aug_root_dir, transform=transformations)
            augmented_datasets.append(augmented_dataset)

        # Concatenate all datasets
        all_datasets = [ori_fl, ori_nf] + augmented_datasets
        train_set = ConcatDataset(all_datasets)


    val_set = MyJP2Dataset(csv_file = csv_file_val, 
                                root_dir = original_image_dir,
                                transform = transformations)

    # test_set = MyJP2Dataset(csv_file = csv_file_test, 
    #                             root_dir = original_image_dir,
    #                             transform = transformations)

    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, num_workers=16, pin_memory = True, shuffle = True)

    val_loader = DataLoader(dataset=val_set, batch_size=100, num_workers=16, pin_memory = True, shuffle = False)
    # test_loader = DataLoader(dataset=test_set, batch_size=opt.batch_size, num_workers=16, pin_memory = True, shuffle = False)

    return train_loader, val_loader


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    #print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train():
    train_loader, val_loader = dataloading()
    # visualize_batch_sample(train_loader)

    #Model Configuration for two GPUs
    learning_rate = opt.learning_rate
    num_epochs = opt.epochs
    # attention_config =opt.attn_config
    alpha = opt.alpha
    weight_decay = opt.weight_decay
    # gamma = opt.gamma
    batch_size = opt.batch_size

    device_ids = [0, 1]
    if opt.models=="mobilenet":
        net = MobileNet().to(device)
        print("Model Selected: ", opt.models)
        # print(net)
    elif opt.models=="mobilevit":
        net = MobileViT(image_size = (512, 512), dims = [96, 120, 144], channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384], num_classes = 2).to(device)
        print("Model Selected: ", opt.models)
        # print(net)
    elif opt.models=="resnet":
        net = Custom_ResNet34(ipt_size=(512, 512), pretrained=True).to(device)
        print("Model Selected: ", opt.models)
        # print(net)
    else:
        print('Invalid Model Selection')

    # net = ViT(image_size=512, patch_size=32, num_classes=2, dim=1024, depth=8, heads=16, mlp_dim=1024, emb_dropout=0.25, dropout=0.25).to(device)
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    count_parameters(model)



    criterion = OrdinalCrossEntropyLoss(alpha=alpha).to(device)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, 
    #                 max_lr = opt.learning_rate, # Upper learning rate boundaries in the cycle for each parameter group
    #                 steps_per_epoch = len(train_loader), # The number of steps per epoch to train for.
    #                 epochs = num_epochs-1, # The number of epochs to train for.
    #                 anneal_strategy = 'cos')
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=2)
    

    # Training Network
    # wandb.watch(model, criterion, log="all", log_freq=10)
    print(f"Training in Progress Model {opt.models}...")
    train_loss_values = []
    val_loss_values = []
    train_tss_values = []
    val_tss_values = []
    train_hss_values = []
    val_hss_values = []
    train_geomean_values = []
    val_geomean_values = []
    train_time = []
    val_time = []
    learning_rate_values = []
    best_geomean = 0
    best_hss = 0
    best_tss = 0
    # best_hss = 0
    for epoch in range(1,num_epochs):

        # #Dynamically Reducing Learning Rate
        # if epoch%5==0:
        #     for g in optimizer.param_groups:
        #         g['lr'] = g['lr']/5
        
        #Timer for Training one epoch
        start_train = timeit.default_timer() 
        
        # setting the model to train mode
        model.train()
        train_loss = 0
        train_tss = 0.
        train_hss = 0.
        train_geomean = 0.
        train_prediction_list = []
        train_target_list = []
        for batch_idx, (data, targets, log_scale) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)
            log_scale = log_scale.to(device=device)
            train_target_list.append(targets)
            
            # forward prop
            scores = model(data)
            loss = criterion(scores, targets, log_scale)
            _, predictions = torch.max(scores,1)
            train_prediction_list.append(predictions)
            
            # backward prop
            optimizer.zero_grad()
            loss.backward()
            
            # SGD step
            optimizer.step()
            # scheduler.step()
            
            # accumulate the training loss
            #print(loss.item())
            train_loss += loss.item()
            #train_acc+= acc.item()
            
        stop_train = timeit.default_timer()
        print(stop_train-start_train)
        train_time.append(stop_train-start_train)

        # Validation: setting the model to eval mode
        model.eval()
        start_val = timeit.default_timer()
        val_loss = 0.
        val_tss = 0.
        val_hss = 0.
        val_geomean = 0.
        val_prediction_list = []
        val_target_list = []

        # Turning off gradients for validation
        with torch.no_grad():
            for d, t, lg_scl in val_loader:
                # Get data to cuda if possible
                d = d.to(device=device)
                t = t.to(device=device)
                lg_scl = lg_scl.to(device=device)
                val_target_list.append(t)
                
                # forward pass
                s = model(d)
                #print("scores", s)
                                    
                # validation batch loss and accuracy
                l = criterion(s, t, lg_scl)
                _, p = torch.max(s,1)
                #print("------------------------------------------------")
                #print(torch.max(s,1))
                #print('final', p)
                val_prediction_list.append(p)
                
                # accumulating the val_loss and accuracy
                val_loss += l.item()
                #val_acc += acc.item()
                del d,t,s,l,p, lg_scl
                torch.cuda.empty_cache()
        stop_val = timeit.default_timer()
        val_time.append(stop_val-start_val)
        learning_rate_values.append(optimizer.param_groups[0]['lr'])

        #Epoch Results
        train_loss /= len(train_loader)
        train_loss_values.append(train_loss)
        val_loss /= len(val_loader)
        val_loss_values.append(val_loss)
        train_tss, train_hss, train_geomean = sklearn_Compatible_preds_and_targets(train_prediction_list, train_target_list)
        train_tss_values.append(train_tss)
        train_hss_values.append(train_hss)
        train_geomean_values.append(train_geomean)
        val_tss, val_hss, val_geomean = sklearn_Compatible_preds_and_targets(val_prediction_list, val_target_list)
        val_tss_values.append(val_tss)
        val_hss_values.append(val_hss)
        val_geomean_values.append(val_geomean)
        scheduler.step(val_geomean)
        if (val_geomean > best_geomean):
            print('New best model found!')
            best_geomean = val_geomean
            best_hss = val_hss
            best_tss = val_tss
            best_epoch = epoch
            best_model = model.module.state_dict()
            best_path = f'{model_dir}/best_loss_{opt.loss}.pth'

        print(f'Epoch: {epoch}/{num_epochs-1}')
        print(f'Training--> loss: {train_loss:.4f}, TSS: {train_tss:.4f}, HSS2: {train_hss:.4f} | Val--> loss: {val_loss:.4f}, TSS: {val_tss:.4f} | HSS2: {val_hss:.4f} ')
        # W&B logging for each epoch
        wandb.log({
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Train TSS": train_tss,
            "Validation TSS": val_tss,
            "Train HSS": train_hss,
            "Validation HSS": val_hss,
            "Validation Mean": val_geomean,
            "Train Mean": train_geomean,
            "Learning Rate": optimizer.param_groups[0]['lr']
        })
    PATH = f'{model_dir}/Epoch_{epoch}_loss{opt.loss}.pth'
    print("**********************************************************************************")
    print("Best Model Selected", 'TSS: ', best_tss, "HSS: ", best_hss, 'Epoch: ', best_epoch)
    print("**********************************************************************************")
    torch.save({
                'epoch': num_epochs,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, PATH)
    torch.save({
                'model_state_dict': best_model
                }, best_path)

    results = {
        'train_tss_values':train_tss_values,
        'val_tss_values':val_tss_values,
        'train_hss_values':train_hss_values,
        'val_hss_values':val_hss_values,
        'train_loss_values':train_loss_values,
        'val_loss_values':val_loss_values,
        'train_geomean_values':train_geomean_values,
        'val_geomean_values':val_geomean_values,
        'learning_rate': learning_rate_values,
        'train_time': train_time,
        'val_time': val_time
    }
    df = pd.DataFrame(results, columns=['train_tss_values','val_tss_values', 'train_hss_values', 'val_hss_values', 'train_loss_values', 'val_loss_values', 'train_geomean_values', 'val_geomean_values', 'learning_rate', 'train_time', 'val_time' ])
    df.to_csv(f'{results_dir}/{opt.loss}.csv', index=False, header=True)



if __name__ == "__main__":
    train()
