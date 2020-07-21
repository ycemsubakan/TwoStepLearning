import numpy as np
import torch
from algorithms_v2 import VAE, conv_autoenc, conv_VAE, get_embeddings, get_scores
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
from torchvision import datasets, transforms
import argparse
import gmms.gmm_learn as cgmm
import torchvision

vis = visdom.Visdom(port=5800, server='http://', env='')
assert vis.check_connection()

# now get (generate) some data and fit 
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_gpus', type=int, help='number of gpus', default=1)
argparser.add_argument('--model', type=str, help='choose your model', default='NF')
arguments = argparser.parse_args()

np.random.seed(2)
torch.manual_seed(9)
arguments.cuda = torch.cuda.is_available()
arguments.batch_size = 128
arguments.data = 'celeba'
arguments.input_type = 'autoenc'

isCrop = 0
if isCrop:
    transform = transforms.Compose([
        transforms.Scale(108),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
else:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
train_data_dir = '/network/tmp1/y/crop_celeba_train/'          # this path depends on your directories
test_data_dir = '/network/tmp1/y/crop_celeba_test/'

dset_train = datasets.ImageFolder(train_data_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=48, shuffle=True,
                                           pin_memory=True, num_workers=arguments.num_gpus)

dset_test = datasets.ImageFolder(test_data_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=24, shuffle=False)

for dt in it.islice(train_loader, 0, 1, 1):
    vis.images(0.5+(dt[0]*0.5), nrow=8, win='celebafaces')
     
    #pdb.set_trace()

#h = torch.randn(100, 100, 1, 1)
#out = netG.forward(Variable(h))

compute_kdes = 1
results = []
mmds = []
fids = []
model = arguments.model

if model == 'VAE': 
    EP = 25
    Kss = [[100, 100]]
    L = 64*64*3
    for config_num, Ks in enumerate(Kss):
        
        mdl = conv_VAE(L, L, Ks, M=64, num_gpus=arguments.num_gpus, arguments=arguments) 
       
        if arguments.cuda:
            mdl.cuda()

        path = 'models/VAE_{}_K_{}.t'.format(arguments.data, Ks)
        if 0 & os.path.exists(path):
            mdl.load_state_dict(torch.load(path))
        else:
            if not os.path.exists('models'):
                os.mkdir('models')
            mdl.VAE_trainer(arguments.cuda, train_loader, vis=vis, 
                            EP=EP, config_num=config_num)
            torch.save(mdl.state_dict(), path)

        #gen_data, seed = mdl.generate_data(100)
        #opts = {'title':'VAE generated data config {}'.format(config_num)}
        #vis.images(0.5 + 0.5*gen_data.data.cpu(), opts=opts, win='VAE_config_{}'.format(config_num))
        
        if compute_kdes:
            num_samples = 1

            scores_iml, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                              num_samples=num_samples,
                                              task='celeba', base_dist='fixed_iso_gauss')
            vis.images(gen_data.data*0.5 + 0.5, win='VAE genim')
            #vis.image(im_gen*0.5 + 0.5, win='VAE genim') 
            #vis.image(im_test*0.5 + 0.5, win='VAE testim') 

elif model == 'VaDE_random_init':
    EP = 5
    num_samples = 1
    Kss = [[100, 100]]
    L = 64*64*3

    for config_num, Ks in enumerate(Kss):
        print(Ks)
        mdl = conv_VAE(L, L, Ks, M=64, num_gpus=arguments.num_gpus, arguments=arguments) 

        mdl.initialize_GMMparams(GMM=None, mode='random')
                    
        if arguments.cuda:
            mdl = mdl.cuda()
        mdl.VAE_trainer_mog(arguments.cuda, train_loader, vis=vis, 
                            EP=EP, config_num=config_num, data='celeba')


elif model == 'VaDE':

    EP = 5
    num_samples = 1
    Kss = [[100, 100]]
    L = 64*64*3
    
    for config_num, Ks in enumerate(Kss):
        print(Ks)
        mdl = conv_VAE(L, L, Ks, M=64, num_gpus=arguments.num_gpus, arguments=arguments) 

        if arguments.cuda:
            mdl.cuda()

        path = 'models/VAE_{}_K_{}.t'.format(arguments.data, Ks)
        if 1 & os.path.exists(path):
            mdl.load_state_dict(torch.load(path))
        else:
            raise ValueError('You have to train the standard Gaussian VAE first, call this script with --model VAE')
            
        all_hhats = get_embeddings(mdl, train_loader)
        all_hhats_chunks = torch.chunk(all_hhats, chunks=35, dim=0)

        if 1:
            scores_vae, samples_vae = get_scores(test_loader, mdl, arguments.cuda, 
                                                 num_samples=num_samples,
                                                 task='celeba', base_dist='fixed_iso_gauss')
            
        use_gmms = [1, 1, 0]
        scores_2step_all = []
        for Kcomps in range(50, 51, 5):
            print('Number of GMM components {}'.format(Kcomps))
            path2 = 'models/VaDEGMM_{}_K_{}_Kmog_{}.t'.format(arguments.data, Ks, Kcomps)
            
            mdl = conv_VAE(L, L, Ks, M=64, num_gpus=arguments.num_gpus, arguments=arguments) 
            if arguments.cuda:
                mdl.cuda()
            mdl.load_state_dict(torch.load(path))

            # train the base distribution 
            if 0 & os.path.exists(path2 + 'Kmog{}.gmm'.format(Kcomps)):
                GMM = pickle.load(open(path2 + 'Kmog{}.gmm'.format(Kcomps), 'rb'))
            else:
                if use_gmms[0]:
                    GMM = mix.GaussianMixture(n_components=Kcomps, verbose=1, n_init=3, max_iter=200, covariance_type='diag')
                    GMM.fit(all_hhats.data.cpu().numpy())
                    pickle.dump(GMM, open(path2 + 'Kmog{}.gmm'.format(Kcomps), 'wb'))
                    mdl.initialize_GMMparams(GMM=GMM)
                    
                if use_gmms[1]:
                    BGMM = mix.GaussianMixture(n_components=Kcomps, verbose=1, n_init=3, max_iter=200, covariance_type='full')
                    BGMM.fit(all_hhats.data.cpu().numpy())
                    pickle.dump(BGMM, open(path2 + 'Kmog{}.bgmm'.format(Kcomps), 'wb'))
                    mdl.GMM = BGMM

                if use_gmms[2]:
                    cudagmm = cgmm.gmm(num_components=Kcomps, L=Ks[0], cuda=arguments.cuda)
                    cudagmm.kmeanspp(all_hhats)
                    cudagmm.kmeans(all_hhats_chunks, vis)
                    cudagmm.em(all_hhats_chunks, em_iters=15)
                    mdl.GMM = cudagmm


            if arguments.cuda:
                mdl.cuda()
           
            if 1:
                if use_gmms[0]:
                    scores_2step, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                                  num_samples=num_samples,
                                                  task='celeba', base_dist='mog')
                else: 
                    scores_2step = []

                if use_gmms[1]:
                    scores_2step_bgmm, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                                  num_samples=num_samples,
                                                  task='celeba', base_dist='mog_skt')

                else:
                    scores_2step_bgmm = []

                if use_gmms[2]:
                    scores_2step_cuda, gen_data = get_scores(test_loader, mdl, arguments.cuda, 
                                                  num_samples=num_samples,
                                                  task='celeba', base_dist='mog_cuda')
                else:
                    scores_2step_cuda = []


                scores_2step_all.append({'Kcomps':Kcomps, 
                                         'scores_iml': scores_2step, 
                                         'scores_iml_bgmm': scores_2step_bgmm,
                                         'score_iml_cuda': scores_2step_cuda})

            
            # now see how the joint training initialized with 2-step training works 
            if 1 & os.path.exists(path2):
                mdl.load_state_dict(torch.load(path2))
                
                scores_vade, gen_data_vade = get_scores(test_loader, mdl, arguments.cuda, 
                                                  num_samples=num_samples,
                                                  task='celeba', base_dist='mog')


            else:
                mdl.VAE_trainer_mog(arguments.cuda, train_loader, vis=vis, 
                                    EP=EP, config_num=config_num, data='celeba')

                torch.save(mdl.state_dict(), path2)

                scores_vade, gen_data_vade = get_scores(test_loader, mdl, arguments.cuda, 
                                                  num_samples=num_samples,
                                                  task='celeba', base_dist='mog')


            results.append({'Kcomps': Kcomps, 
                            'scores_2step': scores_2step, 
                            'scores_2step_bgmm': scores_2step_bgmm,
                            'scores_vade': scores_vade})
            pdb.set_trace()








