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
from sklearn.decomposition import NMF

vis = visdom.Visdom(port=5800, server='http://', env='')
assert vis.check_connection()

# now get (generate) some data and fit 
np.random.seed(2)
torch.manual_seed(9)

argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=2)
argparser.add_argument('--model', type=str, help='this chooses the model [VAE, VaDE]', default='VAE')
argparser.add_argument('--batch_size', type=int, help='batchsize', default=3000)
argparser.add_argument('--get_trfid', type=int, help='if this is 1, we compute training fids also, the value is either 0 or 1', default=1)
arguments = argparser.parse_args()

arguments.cuda = torch.cuda.is_available()
arguments.data = 'mnist'
arguments.input_type = 'autoenc'

train_loader, test_loader = ut.get_loaders(arguments.batch_size, c=0, 
                                           arguments=arguments)

L1 = L2 = 784
M = N = 28

results = []
results_impl = []
Kss = [[K, 600] for K in range(20, 21, 20)]  
model = arguments.model
num_samples = 3

base_path = os.path.expanduser('~') + '/Dropbox'  # you might need to change this
model_path = base_path + '/two_step_learning/models'

if model == 'VAE': 
    EP = 1250 
    for config_num, Ks in enumerate(Kss):
        print(Ks)
        mdl = VAE(L1, L2, Ks, M=M, arguments=arguments) 
       
        if arguments.cuda:
            mdl.cuda()

        path = model_path + '/VAE_{}_K_{}.t'.format(arguments.data, Ks)
        if os.path.exists(path):
            mdl.load_state_dict(torch.load(path))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')


            # train the VAE
            mdl.VAE_trainer(arguments.cuda, train_loader, vis=vis, 
                            EP=EP, config_num=config_num)

            torch.save(mdl.state_dict(), path)

        scores_vae, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                          num_samples=num_samples,
                                          task='mnist', base_dist='fixed_iso_gauss')

            
elif model == 'VaDE':

    EP = 100 
    num_samples = 1
    
    for config_num, Ks in enumerate(Kss):
        print(Ks)
        mdl = VAE(L1, L2, Ks, M=M, arguments=arguments) 
       
        if arguments.cuda:
            mdl.cuda()

        path = 'models/VAE_{}_K_{}.t'.format(arguments.data, Ks)
        if 1 & os.path.exists(path):
            mdl.load_state_dict(torch.load(path))
        else:
            raise ValueError('You have to train a VAE first, please set --model VAE, and train a standard Gaussian VAE')
        all_hhats = get_embeddings(mdl, train_loader)
        all_hhats_test = get_embeddings(mdl, test_loader)

        scores_vae, vae_samples = get_scores(test_loader, mdl, arguments.cuda, 
                                   num_samples=num_samples,
                                   task='mnist', base_dist='fixed_iso_gauss')

        if arguments.get_trfid:
            scores_vae_tr, _ = get_scores(train_loader, mdl, arguments.cuda, 
                                          num_samples=num_samples,
                                          task='mnist', base_dist='fixed_iso_gauss')

            
        for Kcomps in range(50, 51, 5):
            print('Number of GMM components {}'.format(Kcomps))
            path2 = 'models/VaDEGMM_{}_K_{}_Kmog_{}.t'.format(arguments.data, Ks, Kcomps)
            
            if 0 & os.path.exists(path2 + 'Kmog{}.gmm'.format(Kcomps)):
                GMM = pickle.load(open(path2 + 'Kmog{}.gmm'.format(Kcomps), 'rb'))
            else:
                GMM = mix.GaussianMixture(n_components=Kcomps, verbose=1, n_init=10, max_iter=200, covariance_type='diag', warm_start=True)
                GMM.fit(all_hhats.data.cpu().numpy())
                pickle.dump(GMM, open(path2 + 'Kmog{}.gmm'.format(Kcomps), 'wb'))

                mean_ll_tr = GMM.score_samples(all_hhats.data.cpu().numpy()).mean()
                mean_ll_test = GMM.score_samples(all_hhats_test.data.cpu().numpy()).mean()



            mdl.initialize_GMMparams(GMM)
            if arguments.cuda:
                mdl.cuda()
           
            scores_2step, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                                num_samples=num_samples,
                                                task='mnist', base_dist='mog')

            if arguments.get_trfid: 
                scores_2step_tr, _ = get_scores(train_loader, mdl, arguments.cuda, 
                                                num_samples=num_samples,
                                                task='mnist', base_dist='mog')

            
            if 1 & os.path.exists(path2):
                mdl.load_state_dict(torch.load(path2))
            else:
                mdl.VAE_trainer_mog(arguments.cuda, train_loader, vis=vis, 
                                    EP=EP, config_num=config_num)

                torch.save(mdl.state_dict(), path2)

            scores_vade, gen_data_vade = get_scores(test_loader, mdl, arguments.cuda, 
                                              num_samples=num_samples,
                                              task='mnist', base_dist='mog')

            if arguments.get_trfid:
                scores_vade_tr, _ = get_scores(train_loader, mdl, arguments.cuda, 
                                               num_samples=num_samples,
                                               task='mnist', base_dist='mog')

            results.append({'Kcomps' : Kcomps, 
                            'scores_2step': scores_2step, 
                            'scores_vade': scores_vade,
                            'scores_2step_tr' : scores_2step_tr,
                            'scores_vade_tr' : scores_vade_tr})





