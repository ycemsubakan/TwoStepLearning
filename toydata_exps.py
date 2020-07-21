import numpy as np
import torch
from algorithms_v2 import VAE, compute_nparam_density, get_embeddings, get_scores
import pdb
from torch.autograd import Variable
import matplotlib.pyplot as plt
import utils as ut
import os
from drawnow import drawnow, figure
import visdom 
import torch.optim.lr_scheduler as tol
import torch.nn.functional as F
import itertools as it
import torch.nn as nn
import sklearn.mixture as mix
import pickle
import argparse
import torchvision
import torch.utils.data as data_utils

def get_loader(numpeaks, batch_size, dataset_type, path = 'gaussian_toy_files/'):
    # num peaks is between 1 and 16 for the gaussian_toy dataset

    sets = torch.load(path + dataset_type +  '.t')
    dataset = data_utils.ConcatDataset([sets[dg] for dg in range(numpeaks)])
    

    loader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                   shuffle=False)
    return loader



vis = visdom.Visdom(port=5800, server='http://@', env='')
assert vis.check_connection()

# now get (generate) some data and fit 
np.random.seed(2)
torch.manual_seed(9)

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
argparser.add_argument('--model', type=str, help='this chooses the model [VAE, VAEpGMM]', default='VAE')
argparser.add_argument('--batch_size', type=int, help='batchsize', default=3000)
argparser.add_argument('--get_trfid', type=int, help='if this is 1, we compute training fids also, the value is either 0 or 1', default=1)
arguments = argparser.parse_args()

arguments.cuda = torch.cuda.is_available()
arguments.data = 'gaussian_toy'

train_loader = get_loader(16, 10000, 'train')
L1 = L2 = 2 # number of input and output dimensions

Ks = [5, 600]

base_path = os.path.expanduser('~') + '/Dropbox'
model_path = base_path + '/two_step_learning/models'

if arguments.model == 'VAE': 
    EP = 1250 
    print(Ks)
    mdl = VAE(L1, L2, Ks, M=2, outlin='linear', toy_data=True, arguments=arguments) 
   
    if arguments.cuda:
        mdl.cuda()

    path = model_path + '/VAE_{}_K_{}.t'.format(arguments.data, Ks)
    if os.path.exists(path):
        mdl.load_state_dict(torch.load(path))
    else:
        if not os.path.exists('models'):
            os.mkdir('models')

        # train the VAE
        mdl.VAE_trainer(arguments.cuda, train_loader, toy_data=True, vis=vis, 
                        EP=EP, config_num=0)

        torch.save(mdl.state_dict(), path)

    samples, _ = mdl.generate_data(1000, base_dist='fixed_iso_gauss')
    samples = samples.data.cpu().numpy()

elif arguments.model == 'VaDE':
    Kcomps = 16
    print('Number of GMM components {}'.format(Kcomps))
    path = 'models/VaDEGMM_{}_K_{}_Kmog_{}.t'.format(arguments.data, Ks, Kcomps)

    mdl = VAE(L1, L2, Ks, M=2, outlin='linear', toy_data=True, arguments=arguments) 
    mdl.initialize_GMMparams(mode='random')

    if arguments.cuda:
        mdl.cuda()
   
    EP = 1250
    if 1 & os.path.exists(path):
        mdl.load_state_dict(torch.load(path))
    else:
        mdl.VAE_trainer_mog(arguments.cuda, train_loader, vis=vis, 
                            EP=EP, config_num=0)

        torch.save(mdl.state_dict(), path)

    samples, _ = mdl.generate_data(1000, base_dist='mog')
    samples = samples.data.cpu().numpy()

# plot before 2-step learning
fs = 16
plt.figure(figsize=(16, 6), dpi=100)

plt.subplot(121) 
plt.plot(samples[:, 0], samples[:, 1], 'o')
plt.title('Samples with joint training, model {} '.format(arguments.model), fontsize=fs)
plt.axis('equal')

# now fit the GMM on the latents
all_hhats = get_embeddings(mdl, train_loader, cuda=False)
Kcomps=16
GMM = mix.GaussianMixture(n_components=Kcomps, verbose=1, n_init=10, max_iter=200, covariance_type='diag', warm_start=True)
GMM.fit(all_hhats.data.cpu().numpy())

mdl.initialize_GMMparams(GMM)

samples, _ = mdl.generate_data(1000, base_dist='mog')
samples = samples.data.cpu().numpy()

# plot what happens after fitting the GMM on the latents
plt.subplot(122) 
plt.plot(samples[:, 0], samples[:, 1], 'o')
plt.title('Samples after fitting the GMM', fontsize=fs)
plt.axis('equal')

plt.savefig(arguments.model + '_GMMfitting.png', format='png') 


