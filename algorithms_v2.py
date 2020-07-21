import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import pdb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch.utils.data as data_utils
import torch.nn.utils.rnn as rnn_utils
from itertools import islice
import os
import torch.nn.init as torchinit
import librosa as lr
import librosa.display as lrd
import utils as ut
import itertools as it
import copy
import visdom
#import pytorch_fft.fft.autograd as pfft
import scipy.linalg as spl
import math
import sklearn.mixture as mix
import string as st
from hmmlearn import hmm
import scipy as sp
import torchvision
import gmms.gmm_learn as gmm


def logsumexp(h, dim=1):
     m, _ = torch.max(h, dim=dim, keepdim=True)
     first_term = (h - m).exp().sum(dim, keepdim=True).log() 

     return first_term + m 

def log_Normal_diag(x, mus, vrs):

    log_normal_all = -0.5 * ( vrs.unsqueeze(0).log() + (torch.pow(x.unsqueeze(1) - mus.unsqueeze(0), 2) / vrs.unsqueeze(0)) ).sum(-1)
    
    return log_normal_all

def log_Normal_standard(x, average=False, dim=None):
    log_normal = -0.5 * torch.pow( x , 2 )
    if average:
        return torch.mean( log_normal, dim )
    else:
        return torch.sum( log_normal, dim )


class VAE(nn.Module):
    def __init__(self, L1, L2, Ks, M, arguments=None, outlin='sigmoid', toy_data=False):
        super(VAE, self).__init__()

        self.arguments = arguments
        self.L1 = L1
        self.L2 = L2
        self.Ks = Ks
        self.M = M
        self.base_dist = 'fixed_iso_gauss'
        self.outlin = outlin
        self.joint_tr = False
        self.toy_data = toy_data

        self.fc1 = nn.Linear(self.L1, self.Ks[1])
        #initializationhelper(self.fc1, 'relu')

        self.fc21 = nn.Linear(self.Ks[1], self.Ks[0])
        #initializationhelper(self.fc21, 'relu')

        self.fc22 = nn.Linear(self.Ks[1], self.Ks[0])
        #initializationhelper(self.fc22, 'relu')

        self.fc3 = nn.Linear(self.Ks[0], self.Ks[1])
        #initializationhelper(self.fc3, 'relu')

        self.fc4 = nn.Linear(self.Ks[1], self.L2)
        #initializationhelper(self.fc4, 'relu')


            
    def initialize_GMMparams(self, GMM=None, mode='GMMinit'): 
        if mode == 'random':
            Kmog = 50

            self.Kmog = Kmog
            self.mus = nn.Parameter(1*torch.randn(self.Ks[0], Kmog))
            self.sigs = nn.Parameter(torch.ones(self.Ks[0], Kmog))
            self.pis = nn.Parameter(torch.ones(Kmog)/Kmog)
        elif mode == 'GMMinit':
            self.mus = nn.Parameter(torch.from_numpy(GMM.means_).t().float())
            self.sigs = nn.Parameter(torch.from_numpy(GMM.covariances_).t().float())
            self.pis = nn.Parameter(torch.from_numpy(GMM.weights_).float())
            self.Kmog = self.pis.size(0)
        else:
            raise ValueError('What Mode?')



    def encode(self, x):
        h1 = F.tanh(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        z1 = F.tanh(self.fc3(z))
        if self.outlin == 'sigmoid':
            return F.sigmoid(self.fc4(z1))
        else:
            return self.fc4(z1)

    def forward(self, inp):
        #if not (type(inp) == Variable):
        #    inp = Variable(inp[0])

        mu, logvar = self.encode(inp)
        h = self.reparameterize(mu, logvar)

        #print('mean of mu {} variance of mu {}'.format(torch.mean(h).data[0], torch.var(h).data[0]))
        if not (self.arguments.data in ['gaussian_toy', 'mnist']):
            h = h.unsqueeze(-1)
        return self.decode(h), mu, logvar, h

    def criterion(self, recon_x, x, mu, logvar):
        eps = 1e-20
        #criterion = lambda lam, tar: torch.mean(-tar*torch.log(lam+eps) + lam)
        recon_x = recon_x.view(-1, self.L2)
        x = x.view(-1, self.L2)

        crt = lambda xhat, tar: torch.sum(((xhat - tar).abs()), 1)
        #mask = torch.ge(recon_x, 1).float()
        #mask2 = torch.le(recon_x, 0).float()
        #recon_x = mask*(1-eps) + (1-mask)*recon_x
        #recon_x = mask2*eps + (1-mask)*recon_x
        #crt = lambda xhat, tar: -torch.sum(tar*torch.log(xhat+eps) + (1-tar)*torch.log(1-xhat+eps), 1)

        BCE = crt(recon_x, x)
        v = 1
        KLD = -0.5 * torch.sum(1 + logvar - ((mu.pow(2) + logvar.exp())/v), 1)
        # Normalise by same number of elements as in reconstruction
        #KLD = KLD /(x.size(0) * x.size(1))

        return BCE + 0.1*KLD

    def criterion_mog(self, recon_x, x, mu, logvar, beta):
        eps = 1e-20
        recon_x = recon_x.view(-1, self.L2)
        x = x.view(-1, self.L2)
        if self.arguments.data in ['celeba', 'gaussian_toy']:
            crt = lambda xhat, tar: torch.sum(((xhat - tar).abs() ), 1)
        elif self.arguments.data == 'mnist':
            crt = lambda xhat, tar: -torch.sum(tar*torch.log(xhat+eps) + (1-tar)*torch.log(1-xhat+eps), 1)
        else:
            raise ValueError('What Data?')

        BCE = crt(recon_x, x)

        loss1 = -((self.mus.unsqueeze(0) - mu.unsqueeze(-1))**2)/self.sigs.unsqueeze(0) - self.sigs.log().unsqueeze(0) 
        loss2 = -(logvar.unsqueeze(-1) - self.sigs.log().unsqueeze(0)).exp()

        noise = torch.randn(mu.size())
        if self.arguments.cuda:
            noise = noise.cuda()

        zs = mu + noise * (0.5*logvar).exp()
        resp = F.softmax( (-(0.5*(self.mus.unsqueeze(0) - zs.unsqueeze(-1))**2)/self.sigs.unsqueeze(0)).sum(1) - 0.5*self.sigs.log().sum(0).unsqueeze(0) + self.pis.unsqueeze(0).log(), dim=-1)

        term1 = 0.5*((loss1 + loss2).sum(1) * resp).sum(-1)

        term2 = (resp * (torch.log(self.pis).unsqueeze(0) - torch.log(resp+eps))).sum(1)  \
                + 0.5 * (1 + logvar).sum(1)  

        NELBO =  (- term1 - term2) * beta + BCE
        
        return NELBO

        #crt = lambda xhat, tar: torch.sum(((xhat - tar).abs() ), 1)
        #resp = F.softmax((loss1 + self.pis.unsqueeze(0).unsqueeze(0).log()).sum(1), dim=-1)
        
        #resp2 = F.softmax(-((self.mus.unsqueeze(0) - zs.unsqueeze(-1))**2).sum(1), dim=1)
        
    def generate_data(self, N, base_dist='fixed_iso_gauss'):

        if base_dist == 'fixed_iso_gauss':
            seed = torch.randn(N, self.Ks[0]) 
            if next(self.parameters()).is_cuda:
            #self is self.cuda(): 
                seed = seed.cuda()
            seed = Variable(seed)
            gen_data = self.decode(seed)
            return gen_data, seed
        elif base_dist == 'mog':
            clsts = np.random.choice(range(self.Kmog), N, p=self.pis.data.cpu().numpy())
            mus = self.mus[:, clsts]
            randn = torch.randn(mus.size())
            if next(self.parameters()).is_cuda:
                randn = randn.cuda()

            zs = mus + (self.sigs[:, clsts].sqrt())*randn
            gen_data = self.decode(zs.t())
            return gen_data, zs
        elif base_dist == 'mog_skt':
            seed = self.GMM.sample(N)[0]
            seed = torch.from_numpy(seed).float()

            if self.cuda:
                seed = seed.cuda()
            return self.decode(seed), seed
        elif base_dist == 'mog_cuda':
            seed = self.GMM.sample(N)
            return self.decode(seed), seed
        else:
            raise ValueError('what base distribution?')

    
    def VAE_trainer_mog(self, cuda, train_loader, 
                        EP = 400,
                        vis = None, config_num=0, data='mnist', **kwargs):

        if hasattr(kwargs, 'optimizer'):
            optimizer = kwargs['optimizer']
        else:
            optimizer = 'Adam'

        self.train(mode=True)

        L1 = self.L1
        L2 = self.L2
        Ks = self.Ks

        lr = 1e-4
        if optimizer == 'Adam':
            optimizerG = optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.999))
        elif optimizer == 'RMSprop':
            optimizerG = optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            optimizerG = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'LBFGS':
            optimizerG = optim.LBFGS(self.parameters(), lr=lr)

        nbatches = 1400
        for ep in range(EP):
            for i, (tar, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
                if cuda:
                    tar = tar.cuda()

                if self.arguments.data in ['synthetic_sounds']:
                    tar = tar.view(-1, 1, L2).float()
                elif self.arguments.data == 'mnist':
                    tar = tar.view(-1, L2)

                tar = Variable(tar)

                # generator gradient
                self.zero_grad()
                out_g, mu, logvar, h = self.forward(tar)
                
                if self.arguments.data in ['celeba', 'mnist', 'synthetic_sounds']:
                    beta = 1
                else: 
                    beta = 0.1
                err_G = self.criterion_mog(out_g, tar, mu, logvar, beta=beta) #( (ep)+1) /EP)
                err_G = err_G.mean(0)

                err_G.backward(retain_graph=True)

                # step 
                optimizerG.step()
                self.pis.data = self.pis.data.abs() / self.pis.data.abs().sum()

                print('EP [{}/{}], error = {}, batch = [{}/{}], config num {}'.format(ep+1, EP, err_G.item(), 
                                                                            i+1, len(train_loader), config_num))
                                                                       
                if data == 'mnist':
                    per = 10
                elif data == 'celeba':
                    per = 2
                else:
                    per = 50 

                if self.arguments.data == 'gaussian_toy' and (ep % 50 == 0): 
                    vis.scatter(tar, win='target')
                    vis.scatter(out_g, win='output')


                if self.arguments.data == 'celeba' and ((i % 50) == 0):
                    # visdom plots
                    # generate samples 

                    self.eval()
                    N = 64
                    self.train(mode=False)
                    gen_data, seed = self.generate_data(N, base_dist='mog')
                    self.train(mode=True)

                    opts={}
                    opts['title'] = 'Generated Images'
                    vis.images(gen_data.data.cpu()*0.5 + 0.5, opts=opts, win='vade')
            #        
                    opts['title'] = 'Approximations'
                    vis.images(0.5*out_g.data.cpu() + 0.5, opts=opts, win='vae_approximations')
                    opts['title'] = 'Input images'
                    vis.images(tar.data.cpu()*0.5 + 0.5, opts=opts, win='vae_x')


                                    
        return h

    def criterion_jointhmm(self, recon_x, x, mu, logvar, e=0):
        eps = 1e-20
        #criterion = lambda lam, tar: torch.mean(-tar*torch.log(lam+eps) + lam)
        recon_x = recon_x.view(-1, self.L2)
        x = x.view(-1, self.L2)

        crt = lambda xhat, tar: torch.sum(((xhat - tar).abs()), 1)
        
        BCE = crt(recon_x, x)
        v = 1
        noise = torch.randn(mu.shape)
        if self.usecuda: 
            noise = noise.cuda()
        hhat = mu + noise*(0.5*logvar).exp()

        KLD = -self.hmm_alpha_recursion(hhat)[0] 
        if e < 0:
            KLD = 0

        #KLD = -0.5 * torch.sum(1 + logvar - ((mu.pow(2) + logvar.exp())/v), 1)
        # Normalise by same number of elements as in reconstruction
        #KLD = KLD /(x.size(0) * x.size(1))
        return KLD + BCE

    def hmm_alpha_recursion(self, hhat):
        al = torch.ones(self.Kdict) / self.Kdict
        if self.usecuda: 
            al = al.cuda()

        tranmat = F.softmax(self.A, dim=0)
        sigs = F.softplus(self.sigs)
        obs_term = log_Normal_diag(hhat, self.mus, self.sigs) 
        eps = 1e-20
        for t in range(hhat.size(0)):
            
            tran_term = torch.log(eps + torch.matmul(tranmat, al) ) 
            
            logal = tran_term + obs_term[t, :]
            al = F.softmax(logal)
            logsumexp_al = logsumexp(logal, dim=0)

        return logsumexp_al, al



    def VAE_trainer(self, cuda, train_loader, 
                    toy_data = False, EP = 400, 
                    vis = None, config_num=0, **kwargs):

        if hasattr(kwargs, 'optimizer'):
            optimizer = kwargs['optimizer']
        else:
            optimizer = 'Adam'

        self.train(mode=True)

        L1 = self.L1
        L2 = self.L2
        Ks = self.Ks

        lr = 2e-4
        if optimizer == 'Adam':
            optimizerG = optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.999))
        elif optimizer == 'RMSprop':
            optimizerG = optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer == 'SGD':
            optimizerG = optim.SGD(self.parameters(), lr=lr)
        elif optimizer == 'LBFGS':
            optimizerG = optim.LBFGS(self.parameters(), lr=lr)

        nbatches = 1400
        for ep in range(EP):
            for i, data in enumerate(it.islice(train_loader, 0, nbatches, 1)):
                tar = data[0]
                if cuda:
                    tar = tar.cuda()

                if self.arguments.data in ['synthetic_sounds']:
                    tar = tar.view(-1, 1, L2).float()
                elif self.arguments.data == 'mnist':
                    tar = tar.view(-1, L2)

                # generator gradient
                self.zero_grad()
                out_g, mu, logvar, h = self.forward(tar)
                if self.joint_tr:
                    err_G = self.criterion_jointhmm(out_g, tar, mu, logvar, ep) 
                else:
                    err_G = self.criterion(out_g, tar, mu, logvar)
                err_G = err_G.mean(0)

                err_G.backward()

                # step 
                optimizerG.step()
                print('EP [{}/{}], error = {}, batch = [{}/{}], config num {}'.format(ep+1, EP, err_G.item(), 
                                                                                      i+1, len(train_loader), config_num))
                                                                       
                if self.arguments.data == 'gaussian_toy' and (ep % 50 == 0): 
                    vis.scatter(tar, win='target')
                    vis.scatter(out_g, win='output')

                #if (i == 20):
                #    print(self.mus[:2, :2])
                #    # visdom plots
                #    # generate samples 

                #    #self.eval()
                #    #self.train(mode=False)
                #    #gen_data, seed = self.generate_data(30)
                #    #self.train(mode=True)

                #    N = 200
                #    opts = {'title' : 'x'}
                #    vis.line(tar.cpu()[0].squeeze(), opts=opts, win='x')

                #    opts = {'title' : 'xhat'}
                #    vis.line(out_g.cpu()[0].squeeze(), opts=opts, win='xhat')
                #    
                #    vis.heatmap(F.softmax(self.A, dim=1), win='A')
                #    vis.heatmap(F.softplus(self.sigs), win='sigs')
                #    vis.heatmap(self.mus, win='mus')



class conv_VAE(VAE):
    def __init__(self, L1, L2, Ks, M, num_gpus=4, arguments=None):
        super(conv_VAE, self).__init__(L1, L2, Ks, M, arguments=arguments)

        self.fc1 = None
        self.fc21 = None
        self.fc22 = None
        self.fc3 = None
        self.fc4 = None

        self.num_gpus = num_gpus
        K = self.K = Ks[0]

        d = 128
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(K, d*8, 4, 1, 0),
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(d*2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.ConvTranspose2d(d, 3, 4, 2, 1),
            nn.Tanh()
        )
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),

            nn.Conv2d(d, d*2, 4, 2, 1),
            nn.BatchNorm2d(2*d),
            nn.ReLU(True),

            nn.Conv2d(d*2, d*4, 4, 2, 1),
            nn.BatchNorm2d(4*d),
            nn.ReLU(True),

            nn.Conv2d(d*4, d*8, 4, 2, 1),
            nn.BatchNorm2d(8*d),
            nn.ReLU(True),

            nn.Conv2d(d*8, 2*K, 4, 1, 0),
        )

    def encode(self, x):
        x = x.view(-1, 3, 64, 64)
        #nn.parallel.data_parallel(self.enc, inp, range(self.num_gpus))
        h = nn.parallel.data_parallel(self.encoder, x, range(self.num_gpus))
        return (h[:, :self.K, 0, 0]), h[:, self.K:, 0, 0]

    def decode(self, z):
        z = z.contiguous().view(-1, self.K, 1, 1)
        z1 = nn.parallel.data_parallel(self.decoder, z, range(self.num_gpus))
        return z1

    #def criterion(self, recon_x, x, mu, logvar):
    #    eps = 1e-20
    #    recon_x = recon_x.view(-1, self.L2)
    #    x = x.view(-1, self.L2)
    #    crt = lambda xhat, tar: torch.sum(((xhat - tar)**2 ), 1)

    #    BCE = crt(recon_x, x)
    #    v = 1
    #    KLD = -0.5 * torch.sum(1 + logvar - ((mu.pow(2) + logvar.exp())/v), 1)
    #    # Normalise by same number of elements as in reconstruction
    #    # KLD = KLD /(x.size(0) * x.size(1))
    #    return BCE + KLD

    #def generate_data(self, N, base_dist='fixed_iso_gauss'):
    #    #self.train(mode=False)
    #    #self.eval()
    #    seed = torch.randn(N, self.Ks[0]) 
    #    if self is self.cuda(): 
    #        seed = seed.cuda()
    #    seed = Variable(seed)
    #    gen_data = self.decode(seed)

    #    return gen_data, seed


def vis_plot_results(model, inv_f, data, vis, all_costs, ep, arguments):
    #if model.base_dist == 'mixture_full_gauss':
    #    gen_data = Variable(torch.randn(200, model.L2))
    #    seed = Variable(torch.randn(200, model.Ks[0]))
    #else:
    gen_data, seed = model.generate_data(200, base_dist='fixed_iso_gauss')
    if arguments == None:
        datatype = 'images'
    else:
        datatype = arguments.data

    if datatype == 'images':
        N = 64
        M = model.M 
        images = ut.collate_images(gen_data, N=N, L=M)
        sz = 800
        opts={'width':sz, 'height':sz}
        opts['title'] = 'Generated Images'
        hm1 = vis.heatmap(images, opts=opts, win='generated')

        M=data.size(1)
        data_images = ut.collate_images(Variable(data.cpu()), N=N, L=M)
        opts['title'] = 'Training Data'
        hm = vis.heatmap(data_images, opts=opts, win='tr_data')
       
        M=inv_f[0].size(1)
        opts['title'] = 'Mapping to Base Distribution Space'
        if M == model.L2:
            hhat_images = ut.collate_images(inv_f[0], N=N, L=M)
            opts['xmax'] = 2
            opts['xmin'] = -0.2
            hm2 = vis.heatmap(hhat_images, opts=opts, win='hhat')
        elif M == 2:
            hhat = inv_f[0].data.cpu()
            opts['markercolor'] = np.ones(hhat.size(0))*0
            vis.scatter(hhat, opts=opts, win='hhat')
            opts['markercolor'] = np.ones(seed.size(0))*100
            vis.scatter(seed.data.cpu(), opts=opts, win='hhat', update='append')

        M=inv_f[-1].size(1)
        xhat_images = ut.collate_images(inv_f[-1], N=N, L=M)
        opts['title'] = 'Remappings to Observation Space'
        hm3 = vis.heatmap(xhat_images, opts=opts, win='xhat')

        opts['title'] = 'Epoch vs Cost'
        opts['xtype'], opts['ytype'] = 'linear', 'log'
        vis.line(Y=all_costs[:ep+1], X=torch.arange(1,ep+2), opts=opts, win='cost')
    
    elif datatype == 'toy_example':

        opts = {}
        opts['title'] = 'Training Data'
        vis.scatter(data.cpu(), win='x', opts=opts, name='x')

        opts['title'] = 'Generated Data'
        vis.scatter(gen_data.data.cpu(), win='xgen', opts=opts)

        #opts['title'] = 'Training Data Histogram'
        #vis.mesh(data.cpu(), win='x_heat', opts=opts)

        opts['title'] = 'Mappings to Base Distribution Space'
        vis.scatter(inv_f[0].data.cpu(), win='hhat', opts=opts)

        opts['title'] = 'Mapping Back to observation space'
        vis.scatter(inv_f[-1].data.cpu(), win='xhat', opts=opts)

        opts['title'] = 'Epoch vs Cost'
        opts['xtype'], opts['ytype'] = 'linear', 'log'
        vis.line(Y=all_costs[:ep+1], X=torch.arange(1,ep+2), opts=opts, win='cost')

def compute_nparam_density(test_loader, model, sig, cuda, num_samples=1, task='celeba'):

    model = model.eval()
    utils = Normalizing_Flow_utils(0.1, cuda=cuda) 

    all_averages = []
    for n in range(num_samples):
        gen_data, _ = model.generate_data(1000, base_dist=model.base_dist) 
        gen_data = gen_data.view(-1, model.L2).data

        all_kdes = []
        for i, (test_data, _) in enumerate(it.islice(test_loader,0,500,1)):
            print('batch {}'.format(i))
            test_data = test_data.view(-1, model.L2)
            kdes = utils.compute_kde(utils.tocuda(test_data), (gen_data), sig)
            _, kde_inds = torch.max(kdes, dim=0)
            kdes = utils.logsumexp(kdes, dim=0)
            all_kdes.append(kdes.squeeze()) 
        all_averages.append(torch.cat(all_kdes, dim=0).mean(0)[0])

        # again for below
        for i, (test_data, _) in enumerate(it.islice(test_loader,0, 2, 1)):
            print('batch {}'.format(i))
            test_data = test_data.view(-1, model.L2)
            kdes = utils.compute_kde(utils.tocuda(test_data), (gen_data), sig)
            _, kde_inds = torch.max(kdes, dim=0)
            kdes = utils.logsumexp(kdes, dim=0)
            all_kdes.append(kdes.squeeze()) 

    n_images = 24
    ncols = n_images

    if task == 'mnist':
        M = model.M
        im_test = ut.collate_images(Variable(test_data.view(-1, M, M)), n_images, ncols)

        im_gen = gen_data[kde_inds[:n_images], :].view(-1, M, M) 
        im_gen = ut.collate_images(Variable(im_gen).view(-1, M, M), n_images, ncols)
    else: 
        M = 64
        im_test = ut.collate_images_color(Variable(test_data.view(-1, 3, M, M)), n_images, ncols)

        im_gen = gen_data[kde_inds[:n_images], :].view(-1, 3, M, M) 
        im_gen = ut.collate_images_color(Variable(im_gen).view(-1, 3, M, M), n_images, ncols)

    return all_averages, im_gen, im_test

def get_scores(test_loader, model, cuda, num_samples=1, task='celeba', 
               base_dist='fixed_iso_gauss'):

    import pytorch_fid.fid_score as fid
    model = model.eval()
    sig = 1

    all_test = []
    for i, (test_data, _) in enumerate(it.islice(test_loader,0,50,1)):
        print('batch {}'.format(i))
        all_test.append(test_data)

    all_test_cat = torch.cat(all_test, dim=0)
    if cuda:
        all_test_cat = all_test_cat.cuda()

    mmds_lin = []
    mmds_stt = []
    fds_all = []
    fids_all = []

    all_test_cat_flt = all_test_cat.view(-1, model.L2)
    for n in range(num_samples):
        gen_data, _ = model.generate_data(1000, base_dist=base_dist) 
        gen_data_flt = gen_data.view(-1, model.L2).data
        
        #print('computing linear mmd')
        #mmd = compute_mmd(gen_data_flt, all_test_cat_flt, sig, cuda, 
        #                kernel='linear')
        #mmds_lin.append(math.sqrt(mmd))
        
        #print('computing stt mmd')
        #mmd = compute_mmd(gen_data_flt, all_test_cat_flt, sig, cuda, 
        #                kernel='stt')
        #mmds_stt.append(math.sqrt(mmd))

        #print('computing frechet distance')
        #fds_all.append(compute_fd(gen_data_flt, all_test_cat_flt, cuda))
        
        if 1:
            print('computing frechet inception distance')
            if task == 'celeba':
                gen_data_fid = gen_data.data.cpu().numpy()*0.5 + 0.5

                if n == 0:
                    all_test_cat = all_test_cat.cpu().numpy()*0.5 + 0.5
            elif task == 'mnist':
                gen_data_fid = gen_data.data.cpu().view(-1, 1, 28, 28)
                gen_data_fid = gen_data_fid.expand(-1, 3, -1, -1).numpy()

                if n == 0:
                    all_test_cat = all_test_cat.cpu().view(-1, 1, 28, 28)
                    all_test_cat = all_test_cat.expand(-1, 3, -1, -1).numpy()

            fids_all.append(fid.calculate_fig_given_arrays(gen_data_fid, all_test_cat,
                            64, cuda, 2048))

    return {'mmd_lin' : mmds_lin,
            'mmd_stt' : mmds_stt,
            'frechet' : fds_all,
            'frechet_inception' : fids_all}, gen_data
            
def compute_mmd(X, Y, sig, cuda, kernel='linear', biased=True):

    utils = Normalizing_Flow_utils(0.1, cuda=cuda) 
    gamma = 1 / (2 * sig**2)

    XX = torch.matmul(X, X.t())
    XY = torch.matmul(X, Y.t())
    YY = torch.matmul(Y, Y.t())

    X_sqnorms = torch.diag(XX)
    Y_sqnorms = torch.diag(YY)

    m = XX.size(0)
    n = YY.size(0)

    if kernel == 'rbf':
        K_XY = (-gamma * (
                -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        K_XX = (-gamma * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
        K_YY = (-gamma * (
                -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

        logKxymean = utils.logsumexp(K_XY.view(-1), 0) - math.log(m*n)
        logKxxmean = utils.logsumexp(K_XX.view(-1), 0) - math.log(m*m)
        logKyymean = utils.logsumexp(K_YY.view(-1), 0) - math.log(n*n)

        #K_XY = utils.logsumexp(K_XY.view(-1), 0)
        mmd2 = logKxxmean.exp() + logKyymean.exp() - 2*logKxymean.exp()

        #term1 = utils.logsumexp(torch.cat([logKxxmean, logKyymean]),0)
    #logmmd = torch.log( (term1 - logKxymean - math.log(2)).exp() - 1) + logKxymean + math.log(2) 
        #mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    #else:
    #    ps=1
        #mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
        #      + (K_YY.sum() - n) / (n * (n - 1))
        #      - 2 * K_XY.mean())
    elif kernel == 'stt':
        K_XY = -2*XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]
        K_XX = -2*XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]
        K_YY = -2*YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]

        mmd2 = (1/(1 + K_XX)).mean() + (1/(1 + K_YY)).mean() - 2*(1/(1 + K_XY)).mean()

    elif kernel == 'linear':
        mmd2 = XX.mean() + YY.mean() - 2*XY.mean()

    elif kernel == 'tanh':
        mmd2 = F.tanh(XX).mean() + F.tanh(YY).mean() - 2*F.tanh(XY).mean()

    return mmd2

def compute_fd(X, Y, cuda):
    mX = X.mean(0).unsqueeze(0)
    mY = Y.mean(0).unsqueeze(0)
    
    nx = X.size(0)
    ny = Y.size(0)

    mterm = (((mX - mY)**2).sum())

    cX = torch.matmul( (X - mX).t(), (X - mX))/((nx*nx) -1 )
    cY = torch.matmul( (Y-mY).t(), (Y-mY))/((ny*ny) -1)

    #cX[cX < 0] = 0
    #cY[cY < 0] = 0

    matmul = torch.matmul(cX, cY)
    U, S, V = torch.svd(matmul) 
    sqrt = torch.matmul(torch.matmul(U, torch.sqrt(torch.diag(S))), V.t()) 
    fid = mterm + torch.trace(cX + cY - 2*sqrt)
    return fid




def get_embeddings(model, train_loader, cuda=True, flatten=True):
    # get hhats for all batches
    nbatches = 100
    all_hhats = []
    for i, (data, _) in enumerate(it.islice(train_loader, nbatches)):
        if cuda:
            data = data.cuda()

        if flatten: 
            data = data.view(-1, model.L2)

        try:
            hhat, _ = model.encode(data)
        except:
            hhat = model.encode(data)

        all_hhats.append(hhat.data.squeeze())
        print('processing batch {}'.format(i))

    return torch.cat(all_hhats, dim=0)


class conv_autoenc(nn.Module):
    # initializers
    def __init__(self, d=128, K=100, Kdict=30, base_inits=20, num_gpus=1):
        super(conv_autoenc, self).__init__()

        self.usesequential = False #True if num_gpus > 1 else False
        self.num_gpus = num_gpus
        if self.usesequential:
            self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(K, d*8, 4, 1, 0),
                    #nn.BatchNorm2d(8*d),
                    nn.ReLU(True),
                    
                    nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
                    #nn.BatchNorm2d(d*4),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
                    #nn.BatchNorm2d(d*2),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(d*2, d, 4, 2, 1),
                    #nn.BatchNorm2d(d),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(d, 3, 4, 2, 1),
                    nn.Tanh()
                )

        else:
            self.deconv1 = nn.ConvTranspose2d(K, d*8, 4, 1, 0)
            self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
            self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
            self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
            self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1),
            #nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.Conv2d(d, d*2, 4, 2, 1),
            #nn.BatchNorm2d(d*2),
            nn.ReLU(True),

            nn.Conv2d(d*2, d*4, 4, 2, 1),
            #nn.BatchNorm2d(d*4),
            nn.ReLU(True),

            nn.Conv2d(d*4, d*8, 4, 2, 1),
            #nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            
            nn.Conv2d(d*8, K, 4, 1, 0),
            #nn.ReLU(True)
        )
        self.GMM = mix.GaussianMixture(n_components=Kdict, verbose=1, n_init=base_inits, max_iter=200, covariance_type='full', warm_start=True)
        
        self.base_dist = 'mixture_full_gauss'
        self.L2 = 3*64*64

        self.flatten = False
        self.cost_type = 'l2'


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # decoder
    
    def decode(self, inp):
        if self.usesequential:
            x = nn.parallel.data_parallel(self.decoder, inp, range(self.num_gpus))
        else:
            x = F.relu((self.deconv1(inp)))
            x = F.relu((self.deconv2(x)))
            x = F.relu((self.deconv3(x)))
            x = F.relu((self.deconv4(x)))
            x = F.tanh(self.deconv5(x))
        return x

    def encode(self, x):
        h = nn.parallel.data_parallel(self.encoder, x, range(self.num_gpus))
        return h

    def trainer(self, train_loader, vis, EP, cuda, config_num=0,
                regularizer='None', zs_all=None):

        if regularizer == 'None':
            lr = 1e-4
        else:
            lr = 2e-5
        opt = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.999)) 

        nbatches = 25000 
        for ep in range(EP):
            for i, (dt, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
                self.zero_grad()
                if cuda:
                    dt = dt.cuda()
                
                if self.flatten:
                    dt = dt.view(-1, self.L2)

                h = self.encode(Variable(dt))

                xhat = self.decode(h)

                if self.cost_type == 'bernoulli':
                    cost = Variable(dt)*torch.log(xhat) + (1-Variable(dt))*torch.log(1-xhat)
                    cost = -cost.mean()
                elif self.cost_type == 'l1':
                    cost = ((Variable(dt) - xhat).abs()).mean()
                elif self.cost_type == 'l2':
                    cost = ((Variable(dt) - xhat)**2).mean()

                if regularizer is not 'None':

                    zs = torch.max(zs_all[i], 1)[1]

                    means = torch.index_select(self.GMM.means, dim=0, index=zs)
                    icovs = torch.index_select(self.GMM.icovs, dim=0, index=zs)

                    h_cent = h - Variable(means)
                    temp = torch.matmul(h_cent.unsqueeze(1), Variable(icovs)).squeeze()

                    #h_cent_all = h.unsqueeze(1) - Variable(self.GMM.means.unsqueeze(0)) 

                    #all_covs = torch.matmul(h_cent_all.permute(1, 2, 0), h_cent_all.permute(1, 0, 2)*Variable(zs_all[i].t().unsqueeze(-1)))
                    #all_covs = all_covs / Variable(zs_all[i].sum(0).view(-1, 1, 1))

                    #eye = 1e-5*torch.eye(means.size(1))
                    #if self.cuda:
                    #    eye = eye.cuda()
                    #eye = Variable(eye)
                    #    
                    #all_ent = [0.5*torch.svd(cov.squeeze())[1].log().sum() for cov in all_covs]
                    
                    reg = (0.5*(h_cent * temp)).mean() #- sum(all_ent)

                    cost = (cost + reg).mean()
                else:
                    cost = cost.mean()

            
                cost.backward()

                opt.step()

                
                print('EP [{}/{}], batch [{}/{}],  Cost is {}, Learning rate is {}, Config num {}'.format(ep+1, EP, i+1, 
                                                                                                          len(train_loader), cost.data[0], 
                                                                                                          opt.param_groups[0]['lr'], config_num))

                if ( (ep*5 + i) % 100 ) == 0:
                    if isinstance(self, conv_autoenc_mice):
                        im1 = ut.collate_images_rectangular(dt, 16, 4, L1=456, L2=320)
                        vis.heatmap(im1, win='x')

                        im2 = ut.collate_images_rectangular(xhat.data, 16, 4, L1=456, L2=320)
                        vis.heatmap(im2, win='xhat')

                    elif isinstance(self, mlp_autoenc):
                        im1 = ut.collate_images(Variable(dt.view(-1, 28, 28)), 64, 8, L=28)
                        vis.heatmap(im1, win='x')

                        im2 = ut.collate_images(xhat.view(-1, 28, 28), 64, 8, L=28)
                        vis.heatmap(im2, win='xhat')

                        if regularizer is not 'None':
                            seed = self.GMM.sample(300)
                            gen_ims = self.decode(Variable(seed))
                            im3 = ut.collate_images(gen_ims, N=64)
                            vis.heatmap(im3, win='gen_x')

                            if h.size(1) == 2:
                                vis.scatter(seed.cpu(), win='samples from the prior')

                        if h.size(1) == 2:
                            ops = {'title' : 'h'}
                            vis.scatter(h.data.cpu(), win='h')
                        
                    else:
                        vis.images(0.5 + (dt.cpu()*0.5), win='x')
                        vis.images(0.5 + (xhat.data.cpu()*0.5), win='xhat')

                        

                 
    def gmm_trainer(self, train_loader, cuda=True, vis=None):

        # get hhats for all batches
        nbatches = 20
        all_hhats = []
        for i, (data, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
            if cuda:
                data = data.cuda()

            if self.flatten:
                data = data.view(-1, self.L2)

            hhat = self.encoder(Variable(data))
            all_hhats.append(hhat.data.squeeze())
            print(i)

        if 0:
            all_hhats = torch.cat(all_hhats, dim=0)

            if vis is not None:
                vis.heatmap(all_hhats.cpu()[:200].t(), win='hhat', opts = {'title':'hhats for reconstructions'})


            # train base dist
            data = all_hhats.cpu().numpy()
            self.GMM.fit(data)
        else: 
            self.GMM.kmeanspp(all_hhats[0])
            self.GMM.kmeans(all_hhats, vis)

            zs_all = self.GMM.em(all_hhats, em_iters=25)
        
        if 0:
            for (dt, _), zs in zip(train_loader, zs_all):
                N = 64
                dt = dt[:N]

                zs = torch.max(zs[:N], 1)[1]
                means = torch.index_select(self.GMM.means, dim=0, index=zs)
                xhat = self.decoder(Variable(means))

                dt_im = ut.collate_images(Variable(dt), N)
                vis.heatmap(dt_im)

                xhat_im = ut.collate_images(xhat, N)
                vis.heatmap(xhat_im)

        return zs_all
    


    def generate_data(self, N, base_dist='mixture_full_gauss'):
        
        if 0:
            seed = self.GMM.sample(N)[0]
            seed = torch.from_numpy(seed).float()
        else:
            seed = self.GMM.sample(N)

        if not self.flatten:
            seed = seed.view(N, -1, 1, 1)
            
        if self.cuda:
            seed = seed.cuda()
        seed = Variable(seed)
        return self.decode(seed), seed

class mlp_autoenc(conv_autoenc):
# initializers
    def __init__(self, Ks, Kdict=30, base_inits=20, num_gpus=1):
        super(mlp_autoenc, self).__init__()

        self.usesequential = True if num_gpus > 1 else False
        self.num_gpus = num_gpus

        L1 = 784
        self.Ks = Ks

        self.decoder = nn.Sequential(
                nn.Linear(Ks[0], Ks[1]),
                nn.Tanh(),
                
                nn.Linear(Ks[1], L1),
                nn.Sigmoid(),
        )

                
        self.encoder = nn.Sequential(
                nn.Linear(L1, Ks[1]),
                nn.Tanh(),
                
                nn.Linear(Ks[1], Ks[0]),
                #nn.LeakyReLU(),
 
        )
        #self.GMM = mix.GaussianMixture(n_components=Kdict, verbose=1, n_init=base_inits, max_iter = 200, covariance_type='full', warm_start=True, tol=1e-6)
        #self.GMM = mix.BayesianGaussianMixture(n_components=100, verbose=1, max_iter = 200, covariance_type='full', tol=1e-3, n_init=20)

        self.GMM = gmm.gmm(num_components=Kdict, L=Ks[0], cuda=True, n_iters=50) 
        self.base_dist = 'mixture_full_gauss'
        self.L2 = L1
        self.flatten = True
        self.cost_type = 'bernoulli'
        self.M = 28



class conv_autoenc_mice(conv_autoenc):
    # initializers
    def __init__(self, d=128, K=100, Kdict=30, base_inits=20, num_gpus=1):
        super(conv_autoenc_mice, self).__init__()

        self.usesequential = True if num_gpus > 1 else False
        self.num_gpus = num_gpus

        nI = math.ceil(456/5)
        nJ = math.ceil(320/5) 

        d=8

        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(K, d*8, (nI, nJ + 4), 1, 0),
                #nn.BatchNorm2d(8*d),
                nn.ReLU(True),
                
                nn.ConvTranspose2d(d*8, d*4, (nI, nJ), 1, 0),
                #nn.BatchNorm2d(d*4),
                nn.ReLU(True),

                nn.ConvTranspose2d(d*4, d*2, (nI, nJ), 1, 0),
                #nn.BatchNorm2d(d*2),
                nn.ReLU(True),

                nn.ConvTranspose2d(d*2, d, (nI, nJ), 1, 0),
                #nn.BatchNorm2d(d),
                nn.ReLU(True),

                nn.ConvTranspose2d(d, 1, (nI, nJ), 1, 0),
                nn.Tanh()
            )

                
        self.encoder = nn.Sequential(
            nn.Conv2d(1, d, (nI, nJ), 1, 0),
            #nn.BatchNorm2d(d),
            nn.ReLU(True),
            
            nn.Conv2d(d, d*2, (nI, nJ), 1, 0),
            #nn.BatchNorm2d(d*2),
            nn.ReLU(True),

            nn.Conv2d(d*2, d*4, (nI, nJ), 1, 0),
            #nn.BatchNorm2d(d*4),
            nn.ReLU(True),

            nn.Conv2d(d*4, d*8, (nI, nJ), 1, 0),
            #nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            
            nn.Conv2d(d*8, K, (nI, nJ + 4), 1, 0),
            #nn.ReLU(True)
        )
        self.GMM = mix.GaussianMixture(n_components=Kdict, verbose=1, n_init=base_inits, max_iter = 200, covariance_type='full', warm_start=True)
        self.base_dist = 'mixture_full_gauss'
        self.L2 = 456*320
        self.flatten = False
        self.cost_type = 'l2'


class netg_dcgan_par(nn.Module):
    def __init__(self, K, d=128, num_gpus=4):
        super(netg_dcgan_par, self).__init__()
        self.K = K
        self.L2 = 3*64*64
        self.base_dist = 'fixed_iso_gauss'
        self.num_gpus = num_gpus 
        self.Ks = [100]
        self.M = 64

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(K, d*8, 4, 1, 0),
            nn.BatchNorm2d(d*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
            nn.BatchNorm2d(d*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
            nn.BatchNorm2d(d*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(d*2, d, 4, 2, 1),
            nn.BatchNorm2d(d),
            nn.ReLU(True),
            nn.ConvTranspose2d(d, 3, 4, 2, 1),
            nn.Tanh()
        )

    # forward method
    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1,1)
        x = nn.parallel.data_parallel(self.dec, input, range(self.num_gpus))
        return x

    def generate_data(self, N, base_dist='fixed_iso_gauss'):
        seed = torch.randn(N, self.K, 1, 1).cuda()
        seed = Variable(seed)
        return self.forward(seed), seed



class netg_dcgan(nn.Module):
    # initializers
    def __init__(self, K, d=128, num_gpus=4):
        super(netg_dcgan, self).__init__()
        self.K = K
        self.L2 = 3*64*64
        self.base_dist = 'fixed_iso_gauss'
        self.num_gpus = num_gpus 
        self.Ks = [100]
        self.M = 64

        self.deconv1 = nn.ConvTranspose2d(K, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

        #self.dec = nn.Sequential(
        #    nn.ConvTranspose2d(K, d*8, 4, 1, 0),
        #    nn.BatchNorm2d(d*8),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(d*8, d*4, 4, 2, 1),
        #    nn.BatchNorm2d(d*4),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(d*4, d*2, 4, 2, 1),
        #    nn.BatchNorm2d(d*2),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(d*2, d, 4, 2, 1),
        #    nn.BatchNorm2d(d),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(d, 3, 4, 2, 1),
        #    nn.Tanh()
        #)


    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        input = input.view(input.size(0), input.size(1), 1,1)
        #x = nn.parallel.data_parallel(self.dec, input, range(self.num_gpus))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

    def generate_data(self, N, base_dist='fixed_iso_gauss'):
        seed = torch.randn(N, self.K, 1, 1).cuda()
        seed = Variable(seed)
        return self.forward(seed), seed

class netd_dcgan_par(nn.Module):
    # initializers
    def __init__(self, d=128, num_gpus=4):
        super(netd_dcgan_par, self).__init__()

        self.num_gpus = num_gpus

        c = 0.1 
        self.disc = nn.Sequential(
                nn.Conv2d(3, d, 4, 2, 1),
                nn.LeakyReLU(c),
                nn.Conv2d(d, d*2, 4, 2, 1),
                nn.BatchNorm2d(d*2),
                nn.LeakyReLU(c),
                nn.Conv2d(d*2, d*4, 4, 2, 1),
                nn.BatchNorm2d(d*4),
                nn.LeakyReLU(c),
                nn.Conv2d(d*4, d*8, 4, 2, 1),
                nn.BatchNorm2d(d*8),
                nn.LeakyReLU(c),
                nn.Conv2d(d*8, 1, 4, 1, 0),
        )


    # forward method
    def forward(self, input):
       
        #input = input.view(-1, 3, 64, 64)
        x = nn.parallel.data_parallel(self.disc, input, range(self.num_gpus))
        
        return x


class netd_dcgan(nn.Module):
    # initializers
    def __init__(self, d=128, num_gpus=4):
        super(netd_dcgan, self).__init__()

        self.num_gpus = num_gpus

        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)

        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        
        self.conv3 =  nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)

        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)


        #self.disc = nn.Sequential(
        #        nn.Conv2d(3, d, 4, 2, 1),
        #        nn.LeakyReLU(0.2),
        #        nn.Conv2d(d, d*2, 4, 2, 1),
        #        nn.BatchNorm2d(d*2),
        #        nn.LeakyReLU(0.2),
        #        nn.Conv2d(d*2, d*4, 4, 2, 1),
        #        nn.BatchNorm2d(d*4),
        #        nn.LeakyReLU(0.2),
        #        nn.Conv2d(d*4, d*8, 4, 2, 1),
        #        nn.BatchNorm2d(d*8),
        #        nn.LeakyReLU(0.2),
        #        nn.Conv2d(d*8, 1, 4, 1, 0),
        #        nn.Sigmoid()
        #)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
       
        #input = input.view(-1, 3, 64, 64)
        #x = nn.parallel.data_parallel(self.disc, input, range(self.num_gpus))
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = (self.conv5(x))

        return x


def weights_init_seq(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def normal_init(m, mean, std):
    if 0: #isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

    

class CharRNN(nn.Module):
    def __init__(self, L, K1, K2, output_size, n_layers, Kdict=30, base_inits = 1, base_dist='GMM', usecuda=True):
        super(CharRNN, self).__init__()
        self.L = L
        self.K1 = K1
        self.K2 = K2
        self.n_layers = n_layers
        #self.encoder = Variable(torch.eye(input_size).cuda())
        #self.encoder = nn.Embedding(L, K1)
        #self.rnn = ModRNNBase(self.model, input_size, hidden_size, n_layers)
        self.rnn = nn.LSTM(1, K2, n_layers, batch_first=True)
        self.decrnn = nn.LSTMCell(K2, K2)
        #self.decrnn_out = nn.Linear(K2, K2)
        self.declin = nn.Linear(K2, 1)

        self.base_dist = base_dist
        if base_dist == 'mixture_full_gauss':
            self.GMM = mix.GaussianMixture(n_components=Kdict, verbose=1, tol=1e-5, n_init=base_inits, max_iter = 200)
        elif base_dist == 'HMM':
            self.HMM = hmm.GaussianHMM(n_components=Kdict, n_iter=1000,
                                       covariance_type='diag', tol=1e-7, verbose='True')

        
    def decode(self, code, ln):
        inp = (code)        
        zerosmat = torch.zeros(code.size())
        if next(self.parameters()).is_cuda:
            zerosmat = zerosmat.cuda()
        zerosmat = Variable(zerosmat)
        h = (zerosmat, zerosmat)
        outs = []
        for l in range(ln):
            h = self.decrnn(inp, h)
            inp = (h[0])
            outs.append(h[0].unsqueeze(1))
        outs = torch.cat(outs, dim=1)
        #outs = outs.permute([1, 0, 2])
        outs = outs.contiguous().view(-1, self.K2)

        outs = self.declin(outs)
        return outs


    def forward(self, inp):
        batch_size = inp.size(0)
        #encoded = torch.index_select(self.encoder, 0, inpt)
        #encoded = self.encoder(inp)
        inp = inp.unsqueeze(-1).float()
        output, _ = self.rnn(inp)
        code = output.sum(1)

        output = self.decode(code, inp.size(1))
        return output, code

    def trainer(self, train_loader, vis, EP, cuda, config_num=0, path=None):
        opt = torch.optim.Adam(self.parameters(), lr=5e-5) 

        nbatches = 10
        for ep in range(EP):
            for i, (dt, tar, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
                self.zero_grad()

                if cuda:
                    dt = dt.cuda()
                    tar = tar.cuda()

                tar_rsh = tar.unsqueeze(1).float()

                xhat, code = self.forward(Variable(tar_rsh))
                
                #tar_rsh = tar.contiguous().view(-1).float()
                #cost = F.cross_entropy(xhat, Variable(tar_rsh))

                cost = ((Variable(tar_rsh) - xhat)**2).mean()

                cost.backward()

                opt.step()

                print('EP [{}/{}], batch [{}/{}], \
                       Cost is {}, Learning rate is {}, \
                       Config num {}, path {}'.format(ep+1, EP, i+1, 
                                             len(train_loader), 
                                             cost.item(), 
                                             opt.param_groups[0]['lr'], 
                                             config_num, path))
                
            N = 200
            opts = {'title' : 'x'}
            vis.line(tar_rsh.cpu()[0].squeeze(), opts=opts, win='x')

            opts = {'title' : 'xhat'}
            vis.line(xhat.squeeze().data.cpu()[0].squeeze(), opts=opts, win='xhat')

            #recons_x = self.join_chars(dt_rsh) 
            #recons_xhat = self.recons_xhat(xhat)
            #opts = {'title': 'xhat'}
            #vis.text(recons_xhat, win='xhat', opts=opts)
            #opts = {'title': 'x'}
            #vis.text(recons_x, win='x', opts=opts)

    

    def generate_data(self, N, arguments):
        if self.base_dist == 'mixture_full_gauss':
            seed = self.GMM.sample(N)[0]
            seed = torch.from_numpy(seed).float()
        elif self.base_dist == 'HMM':
            seed = self.HMM.sample(N)[0]
            seed = torch.from_numpy(seed).float()
        elif self.base_dist == 'RNN':
            self.training = False
            bs = 2
            #zerosmat = torch.zeros(  ,bs, self.RNN.K)
            #if self.usecuda: #next(self.parameters()).is_cuda:
            #    zerosmat = zerosmat.cuda()
            #zerosmat = Variable(zerosmat)
            #h = (zerosmat, zerosmat)
            inp = torch.zeros(bs, 1, self.RNN.L)
            if self.usecuda:
                inp = inp.cuda()
            inp = Variable(inp)
            outs = []
            for l in range(N):
                if l == 0:
                    h = self.RNN.rnn(inp)
                else: 
                    h = self.RNN.rnn(inp, h[1])

                U = torch.rand(h[0].squeeze().size())
                if self.usecuda:
                    U = U.cuda()
                U = Variable(U)
                lam = 0.001
                sms = self.RNN.sm(h[0].squeeze())
                
                samples = torch.multinomial(F.softmax(sms, dim=1), 1) 

                eye = torch.eye(self.RNN.K)
                if self.usecuda:
                    eye = eye.cuda()
                eye = Variable(eye)

                smh = eye[samples.squeeze()]
                
                #G = - torch.log(-torch.log(U))
                #smh = F.softmax( (sms + G  )/lam, dim=1)

                mu = torch.matmul(self.RNN.out_mean, smh.t()).t()
                if self.RNN.usevar:
                    std = torch.matmul(self.RNN.out_logvar, smh.t()).t()
                else:
                    std = 0.2
                z = torch.randn(bs, self.RNN.L)
                if self.usecuda: 
                    z = z.cuda()
                inp = (Variable(z)*std + mu).unsqueeze(1)
                outs.append(inp) 
            outs = torch.cat(outs, dim=1) 
            seed = outs.view(-1, outs.size(-1), 1)

        if self.base_dist is not 'RNN':
            seed = seed.view(N, -1, 1)
            if self.cuda:
                seed = seed.cuda()
            seed = Variable(seed)
        
        decoded = self.dec(seed)
        return decoded, seed


    def join_chars(self, chnk):
        all_characters = st.printable
        n_characters = len(all_characters)

        recons = ''.join([all_characters[ch] for ch in list(chnk.cpu().numpy())])
        return recons



    def init_hidden(self, batch_size):
        if 'lstm' in self.model:
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


class audionet(VAE):
    def __init__(self, L, K1, K2, output_size, n_layers, Kdict=30, base_inits = 1, base_dist='GMM', num_gpus=1, usecuda=True,
                 joint_tr=False, arguments=None):
        super(audionet, self).__init__(L, L, [K1, K1], L, arguments=arguments) 
                                       
        self.rnn = None
        self.decrnn = None
        self.declin = None

        self.fc1 = None
        self.fc21 = None
        self.fc22 = None
        self.fc3 = None
        self.fc4 = None

        self.num_gpus = num_gpus
        self.usecuda = usecuda

        d = 64
        kernel_size = math.floor(L/4)
        self.kernel_size = kernel_size
        self.K = K1
        self.Kdict = Kdict 

        self.enc = nn.Sequential(
            nn.Conv1d(1, d, kernel_size, 2, int((kernel_size-1)/2)),
            nn.Tanh(),
            nn.Conv1d(d, 2*d, kernel_size, 2, int((kernel_size-1)/2)),
            nn.Tanh(),
            nn.Conv1d(2*d, 2*K1, kernel_size, 1, 0),
            #nn.Tanh(),
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose1d(K1, d, kernel_size, 1, 0),
            nn.Tanh(),
            nn.ConvTranspose1d(d, 2*d, kernel_size, 2, int((kernel_size-1)/2)),
            nn.Tanh(),
            nn.ConvTranspose1d(2*d, 1, kernel_size, 2, int((kernel_size-1)/2)),
            #nn.Tanh()
        )

        self.base_dist = base_dist
        if base_dist == 'mixture_full_gauss':
            self.GMM = mix.GaussianMixture(n_components=Kdict, verbose=1, tol=1e-5, n_init=base_inits, max_iter = 200)
        elif base_dist == 'HMM':
            self.HMM = hmm.GaussianHMM(n_components=Kdict, n_iter=1000,
                                       covariance_type='diag', tol=1e-7, verbose='True')

        self.joint_tr = joint_tr
        if joint_tr: 
            # K1 is the observed dimension for the HMM
            self.A = nn.Parameter(torch.eye(Kdict, Kdict))
            #self.pi = nn.Parameter(torch.ones(Kdict) / Kdict)

            self.mus = nn.Parameter(torch.randn(Kdict, K1)*0.01)
            self.sigs = nn.Parameter(torch.rand(Kdict, K1)*0.01) 

    def encode(self, x):
        #nn.parallel.data_parallel(self.enc, inp, range(self.num_gpus))
        h = self.enc(x)
        return (h[:, :self.K, 0]), h[:, self.K:, 0]

    def decode(self, z):
        z = z.reshape(-1, self.K, 1)
        xhat = self.dec(z) 
        return xhat

    def base_dist_trainer(self, train_loader, cuda=True, vis=None, path=None):
        # get hhats for all batches
        nbatches = None
        all_hhats = []
        for i, (data, _, _) in enumerate(it.islice(train_loader, 0, nbatches, 1)):
            
            if cuda:
                data = data.cuda()
            data = data.unsqueeze(1)
            xhat, mu, logvar, hhat = self.forward(Variable(data.float()))

            all_hhats.append(hhat.squeeze())
            print(i)
        all_hhats = torch.cat(all_hhats, dim=0)
        
        # reconstruct
        xhat_concat = ut.pt_to_audio_overlap(xhat)
        lr.output.write_wav('reconstructions.wav', xhat_concat, 8000, norm=True)
        # see the embeddings
        
        vis.heatmap(all_hhats.data.cpu()[:100].t(), win='hhat', opts = {'title':'hhats for reconstructions'})
        

        #lr.output.write_wav('original_data.wav', data.cpu().numpy().reshape(-1)
        #                    , 8000, norm=True)

        # train base dist
        if self.base_dist == 'mixture_full_gauss':
            data = all_hhats.data.cpu().numpy()
            self.GMM.fit(data)
        elif self.base_dist == 'HMM':
            data = all_hhats.data.cpu().numpy()
            lenseq = 400
            N = data.shape[0] - data.shape[0] % lenseq

            lengths = [lenseq] * (N//lenseq)
            self.HMM.fit(data, lengths)
        elif self.base_dist == 'RNN':
            # get the loader here
            inputs = all_hhats[:-1, :].data.cpu()
            targets = all_hhats[1:, :].data.cpu()

            def form_rnn_dataset(all_hhats):
                split_size = 100
                splits = torch.split(all_hhats, split_size=100, dim=0)
                splits = [splt.unsqueeze(0) for splt in splits if splt.size(0) == 100]  # filter out different size sequences
                concat = torch.cat(splits, dim=0)
                return concat 
            inputs_splt = form_rnn_dataset(inputs).contiguous()
            targets_splt = form_rnn_dataset(targets).contiguous()

            train_dataset = data_utils.TensorDataset(data_tensor=inputs_splt,
                                                     target_tensor=targets_splt)
            loader = data_utils.DataLoader(train_dataset, batch_size=100, shuffle=False, 
                                           pin_memory=True, num_workers=1)

            self.RNN = GaussianRNN(L=self.K1, K=30, usecuda=self.usecuda,
                                   usevar=False)
            if self.usecuda:
                self.RNN = self.RNN.cuda()

            path = path + '.rnn'
            if 1 & os.path.exists(path):
               self.RNN.load_state_dict(torch.load(path)) 
            else:
                muhat = self.RNN.trainer(loader)
                torch.save(self.RNN.state_dict(), path)

                xhat_frames = self.dec(muhat.unsqueeze(-1))
                xhat_concat = ut.pt_to_audio_overlap(xhat_frames)

                lr.output.write_wav('rnn_reconstructions.wav', xhat_concat, 8000, norm=True)

    def generate_data(self, N, arguments):
        if self.base_dist == 'mixture_full_gauss':
            seed = self.GMM.sample(N)[0]
            seed = torch.from_numpy(seed).float()
        elif self.base_dist == 'HMM':
            seed = self.HMM.sample(N)[0]
            seed = torch.from_numpy(seed).float()
        elif self.base_dist == 'RNN':
            self.training = False
            bs = 2
            #zerosmat = torch.zeros(  ,bs, self.RNN.K)
            #if self.usecuda: #next(self.parameters()).is_cuda:
            #    zerosmat = zerosmat.cuda()
            #zerosmat = Variable(zerosmat)
            #h = (zerosmat, zerosmat)
            inp = torch.zeros(bs, 1, self.RNN.L)
            if self.usecuda:
                inp = inp.cuda()
            inp = Variable(inp)
            outs = []
            for l in range(N):
                if l == 0:
                    h = self.RNN.rnn(inp)
                else: 
                    h = self.RNN.rnn(inp, h[1])

                U = torch.rand(h[0].squeeze().size())
                if self.usecuda:
                    U = U.cuda()
                U = Variable(U)
                lam = 0.001
                sms = self.RNN.sm(h[0].squeeze())
                
                samples = torch.multinomial(F.softmax(sms, dim=1), 1) 

                eye = torch.eye(self.RNN.K)
                if self.usecuda:
                    eye = eye.cuda()
                eye = Variable(eye)

                smh = eye[samples.squeeze()]
                
                #G = - torch.log(-torch.log(U))
                #smh = F.softmax( (sms + G  )/lam, dim=1)

                mu = torch.matmul(self.RNN.out_mean, smh.t()).t()
                if self.RNN.usevar:
                    std = torch.matmul(self.RNN.out_logvar, smh.t()).t()
                else:
                    std = 0.2
                z = torch.randn(bs, self.RNN.L)
                if self.usecuda: 
                    z = z.cuda()
                inp = (Variable(z)*std + mu).unsqueeze(1)
                outs.append(inp) 
            outs = torch.cat(outs, dim=1) 
            seed = outs.view(-1, outs.size(-1), 1)

        if self.base_dist is not 'RNN':
            seed = seed.view(N, -1, 1)
            if self.cuda:
                seed = seed.cuda()
            seed = Variable(seed)
        
        decoded = self.dec(seed)
        return decoded, seed

    def np_to_pt_HMM(self):
        self.A = nn.Parameter(torch.from_numpy(self.HMM.transmat_).t()).float().cuda()
        #self.pi = nn.Parameter(torch.ones(Kdict) / Kdict)

        self.mus = nn.Parameter(torch.from_numpy(self.HMM.means_)).float().cuda()
        self.sigs = nn.Parameter(torch.from_numpy(self.HMM.covars_).sum(-1)).float().cuda()

    def pt_to_np_HMM(self):
        transmat = F.softmax(self.A.t(), dim=1).data.cpu().numpy()
        self.HMM.transmat_ = transmat

        mus = self.mus.data.cpu().numpy()
        self.HMM.means_ = mus

        covars = F.softplus(self.sigs)
        
        #all_covs = torch.zeros(covars.size(0), covars.size(1), covars.size(1))
        #for n in range(covars.size(0)):
        #    all_covs[n, :, :] = torch.diag(covars[n, :])  

        self.HMM.covars_ = covars.data.cpu().numpy()
        self.HMM.covariance_type = 'diag'

    
class GaussianRNN(nn.Module):
    def __init__(self, L, K=200, usecuda=False, usevar=False):
        super(GaussianRNN, self).__init__()
        self.L = L
        self.K = K
        self.usecuda = usecuda
        self.num_layers = 1
        self.usevar = usevar

        self.rnn = nn.LSTM(input_size=L, hidden_size=K, num_layers=self.num_layers, batch_first=True, dropout=0.2)
        #self.out_mean = nn.Linear(K, L)
        #torchinit.uniform(self.out_mean.weight, a=-0.01, b=0.01) 
        #torchinit.uniform(self.out_mean.bias, a=-0.1, b=0.1) 
        c = 0.1
        self.sm = nn.Linear(K, K)
        self.out_mean = nn.Parameter(c*torch.randn(L, K))

        if usevar:
            self.out_logvar = nn.Parameter(c*torch.randn(L, K))
        #torchinit.uniform(self.out_logvar.weight, a=-0.01, b=0.01) 
        #torchinit.uniform(self.out_logvar.bias, a=-2, b=0) 

    def forward(self, inp, lam):
        h = self.rnn(inp)[0]
        h_flat = h.contiguous().view(-1, h.size(-1))



        U = torch.rand(h_flat.size())
        if self.usecuda:
            U = U.cuda()
        G = Variable(-torch.log(-torch.log(U)))
        sms = F.softmax((self.sm(h_flat))/lam, dim=1)

        mu = torch.matmul(self.out_mean, sms.t()).t()
        if self.usevar:
            logsig = torch.matmul(self.out_logvar, sms.t()).t()
            return mu, logsig
        else:
            return mu, None

    def trainer(self, train_loader): 
        opt = torch.optim.RMSprop(self.parameters(), lr=1e-3) 
        
        # get a loader here
        EP = 3000
        lam = 1
        for ep in range(EP):
            for i, (dt, tar) in enumerate(train_loader):
                self.zero_grad()

                if self.usecuda:
                    dt = dt.cuda()
                    tar = tar.cuda()
                tar_rsh = tar.view(-1, self.L)
                
                if ep > 0 and ep % 1000 == 0:
                    if self.usevar:
                        lam = lam*0.5
                        print(lam)
                    else:
                        lam = lam*1
                muhat, logsighat = self.forward(Variable(dt), lam)
                
                #logsighat = logsighat.unsqueeze(-1)
                #tar_rsh = tar.contiguous().view(-1).float()
                #cost = F.cross_entropy(xhat, Variable(tar_rsh))

                if self.usevar:
                    eps = 1e-20
                    cost = ( (((Variable(tar_rsh) - muhat)**2)/ (eps + 2*(2*logsighat).exp())).sum(1) + logsighat.sum(1) ).mean()
                    print(logsighat.exp().data.max()) 
                else: 
                    cost = ((Variable(tar_rsh) - muhat)**2).mean()

                #cost = (((self.out_mean.unsqueeze(0) - Variable(tar_rsh).unsqueeze(-1))**2) / ((2*self.out_logvar).exp()*2)).sum(1) + self.out_logvar.sum(0).unsqueeze(0)
                #cost = (cost*sms).sum(1).mean()
                
                #
                cost.backward()

                opt.step()

                print('EP [{}/{}], batch [{}/{}], \
                       Cost is {}, Learning rate is {}'.format(ep+1, EP, i+1, 
                                                         len(train_loader), 
                                                         cost.data[0], 
                                                         opt.param_groups[0]['lr'])) 
        #muhat = torch.matmul(self.out_mean, sms.t()).t()
        return muhat

def elbo_gmm(self, recon_x, x, mu, logvar):
    eps = 1e-20
    recon_x = recon_x.view(-1, self.L2)
    x = x.view(-1, self.L2)
    crt = lambda xhat, tar: -torch.sum(tar*torch.log(xhat+eps) + (1-tar)*torch.log(1-xhat+eps), 1)

    BCE = crt(recon_x, x)

    loss1 = -((self.mus.unsqueeze(0) - mu.unsqueeze(-1))**2)/self.sigs.unsqueeze(0) - self.sigs.log().unsqueeze(0) 
    loss2 = -(logvar.unsqueeze(-1) - self.sigs.log().unsqueeze(0)).exp()

    zs = mu + torch.randn(mu.size()).cuda() * (0.5*logvar).exp()
    resp = F.softmax( (-(0.5*(self.mus.unsqueeze(0) - zs.unsqueeze(-1))**2)/self.sigs.unsqueeze(0)).sum(1) - \
           0.5*self.sigs.log().sum(0).unsqueeze(0) + self.pis.unsqueeze(0).log(), dim=-1)

    term1 = 0.5*((loss1 + loss2).sum(1) * resp).sum(-1)

    term2 = (resp * (torch.log(self.pis).unsqueeze(0) - torch.log(resp+eps))).sum(1)  \
            + 0.5 * (1 + logvar).sum(1)  

    NELBO =  - term1 - term2 + BCE
    
    return NELBO

