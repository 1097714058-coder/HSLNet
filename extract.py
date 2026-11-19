from __future__ import print_function
import argparse
import time
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from NFC import NFC
from clustercontrast.utils import *
import pdb
import scipy.io
import matplotlib.pyplot as plt
from clustercontrast.model_vit_cmrefine.make_model import make_model
from config.defaults import _C

from cluster_contrast_vit_sysu import process_gallery_sysu, process_query_sysu,TestData
import torch
import ast
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data

from config.defaults import _C
from clustercontrast.utils.data import transforms
import os
import argparse
parser = argparse.ArgumentParser(description='PMT Training')
parser.add_argument('--config_file', default='vit_base_ics_288.yml',
                    help='path to config file', type=str)
parser.add_argument('--dataset', default='sysu', help= 'dataset name: regdb or sysu ')
parser.add_argument('--resume', '-r', default='/root/autodl-tmp/project/logs/sysu_train/model_best.pth.tar', type=str,
                    help='resume from checkpoint')
parser.add_argument('--model_path', default='/root/autodl-tmp/project/vit_base_ics_cfs_lup.pth', type=str,
                    help='model save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--test-batch', default=64, type=int,
                    help='testing batch size')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str,
                    help='all or indoor for sysu')
def str_to_list(value):
    return ast.literal_eval(value)

parser.add_argument('--trial', default=1, type=int, help='trial for regdb and LLCM')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'


data_path = '/root/autodl-tmp/project/data/sysu'
n_class = 395
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
print('==> Building model..')
model = make_model(_C, num_class=395, camera_num=6, view_num = 0)
model.cuda()
model.eval()
cudnn.benchmark = True

checkpoint_path = args.model_path

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288,144)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()


def extract_gall_feat(model,gall_loader,ngall):
    pool_dim=768*2
    net = model
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc,feat_fc_s = net( input,input, 1)
            feat_fc = NFC(feat_fc)
            feat_fc_s = NFC(feat_fc_s)
            feat_fc = torch.cat((feat_fc,feat_fc_s),dim=1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1,feat_fc_1_s = net( flip_input,flip_input, 1)
            feat_fc_1 = NFC(feat_fc_1)
            feat_fc_1_s = NFC(feat_fc_1_s)
            feat_fc_1 = torch.cat((feat_fc_1,feat_fc_1_s),dim=1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_query_feat(model,query_loader,nquery):
    pool_dim=768*2
    net = model
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc,feat_fc_s = net( input, input,2)
            feat_fc = NFC(feat_fc)
            feat_fc_s = NFC(feat_fc_s)
            # feat_fc = torch.cat((feat_fc,feat_fc_s),dim=1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1,feat_fc_1_s= net( flip_input,flip_input, 2)
            feat_fc_1 = NFC(feat_fc_1)
            feat_fc_1_s = NFC(feat_fc_1_s)
            # feat_fc_1 = torch.cat((feat_fc_1,feat_fc_1_s),dim=1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            feature_fc_s = (feat_fc_s.detach() + feat_fc_1_s.detach())/2


            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            fnorm_fc_s = torch.norm(feature_fc_s, p=2, dim=1, keepdim=True)
            feature_fc_s = feature_fc_s.div(fnorm_fc_s.expand_as(feature_fc))
            feature_fc = torch.cat((feature_fc,feature_fc_s),dim=1)

            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()

            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc


if len(args.resume) > 0:
    model_path = args.model_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        model.load_param(model_path)
        #model.load_state_dict(model_path, strict=False)
        print(model_path)
        print('==> loaded checkpoint {}'.format(args.resume))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))


    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode='all')
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode='all', trial=1)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(288, 144))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    query_feat_fc = extract_query_feat(model,query_loader,nquery)
    #for i in tqdm(range(10)):
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=1) #all

    trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(288, 144))
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    # query_feat_fc = extract_query_feat(query_loader)
    # gall_feat_fc = extract_gall_feat(trial_gall_loader)

    result = {'gallery_f':gall_feat_fc,'gallery_label':gall_label,'gallery_cam':gall_cam,'query_f':query_feat_fc,'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('tsne.mat',result)