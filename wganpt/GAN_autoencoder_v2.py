import numpy as np
import torch
import os
import pdb
import models.wgan
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
from torch.autograd import Variable
import visdom
import torch.nn as nn
import sys
import torch.nn.functional as F
import sklearn.mixture as mix
import pickle

path = '..'
sys.path.insert(0, path)

from algorithms_v2 import get_embeddings, get_scores
import utils as ut


class netG(nn.Module):
    def __init__(self, L2, Ks, M=28, out='sigmoid'):
        super(netG, self).__init__()
        self.L2 = L2
        self.M = M
        self.Ks = Ks
        self.base_dist = 'iso_fixed_gauss'
        self.out = out

        self.l1 = nn.Linear(self.Ks[0], self.Ks[1], bias=True)
        c = 0.0001
        nn.init.uniform(self.l1.weight, a=-c, b=c)
        nn.init.constant(self.l1.bias, 0)

        self.l2 = nn.Linear(self.Ks[1], self.L2, bias=True)
        nn.init.uniform(self.l2.weight, a=-c, b=c)
        nn.init.constant(self.l2.bias, 0)


    def forward(self, inp): 
        inp = inp.view(-1, self.Ks[0])

        h1 = F.tanh(self.l1(inp))
        if self.out == 'sigmoid':
            output = F.sigmoid(self.l2(h1))
        else:
            output = self.l2(h1)

        return output
    

    def generate_data(self, N, base_dist='fixed_iso_gauss'):

        if base_dist == 'fixed_iso_gauss':
            seed = torch.randn(N, self.Ks[0]) 
            if next(self.parameters()).is_cuda:
            #self is self.cuda(): 
                seed = seed.cuda()
            seed = Variable(seed)
            gen_data = self.forward(seed)
            return gen_data, seed
        elif base_dist == 'mog':
            clsts = np.random.choice(range(self.Kmog), N, p=self.pis.data.cpu().numpy())
            mus = self.mus[:, clsts]
            randn = torch.randn(mus.size())
            if next(self.parameters()).is_cuda:
                randn = randn.cuda()

            zs = mus + (self.sigs[:, clsts].sqrt())*randn
            gen_data = self.forward(zs.t())
            return gen_data, zs
        elif base_dist == 'mog_skt':
            seed = self.GMM.sample(N)[0]
            seed = torch.from_numpy(seed).float()

            if self.cuda:
                seed = seed.cuda()
            return self.forward(seed), seed
        else:
            raise ValueError('what base distribution?')



class encoder(nn.Module):
    def __init__(self, L, Ks, out='sigmoid'):
        super(encoder, self).__init__()
        self.L = self.L2 =  L
        self.Ks = Ks
        self.out = out

        self.l1 = nn.Linear(self.L, self.Ks[1], bias=True)
        c = 0.0001
        nn.init.uniform(self.l1.weight, a=-c, b=c)
        nn.init.constant(self.l1.bias, 0)

        self.l2 = nn.Linear(self.Ks[1], self.Ks[0], bias=True)
        nn.init.uniform(self.l2.weight, a=-c, b=c)
        nn.init.constant(self.l2.bias, 0)


    def encode(self, inp): 
        inp = inp.view(-1, self.L)

        h1 = F.tanh(self.l1(inp))
        output = self.l2(h1)

        return output

home = os.path.expanduser('~')
dec = torch.load('mnist_generator.pt')

vis = visdom.Visdom(port=5800, server='http://', env='')
assert vis.check_connection()

cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda else "cpu")

class args():
    def __init__(self):
        pass

arguments = args()

arguments.batch_size = 3000
arguments.data = 'mnist'
arguments.input_type = 'autoenc'
arguments.cuda = True

dataset_loader, test_loader = ut.get_loaders(arguments.batch_size, c=0, 
                                             arguments=arguments)

# compute the scores here for plain GAN
num_samples = 1
if 1: 
    print('Computing WGAN scores..')
    scores_gan, _ = get_scores(test_loader, dec, arguments.cuda, 
                               num_samples=num_samples,
                               task='mnist', base_dist='fixed_iso_gauss')

enc = encoder(784, [40, 600])
if cuda:
    enc = enc.cuda()
opt = optim.Adam(enc.parameters(), lr=1e-4, betas=(0.5, 0.9)) 

if 0 and os.path.exists('mnist_encoder.pt'):
    enc.load_state_dict(torch.load('mnist_encoder.pt'))
else: 
    for ep in range(100):
        for i, (tar, _) in enumerate(dataset_loader):
            if cuda:
                tar = tar.cuda()

            hhat = enc.encode(tar)
            xhat = dec.forward(hhat)

            cost = ((xhat - tar.reshape(-1, 784))**2).mean() 

            cost.backward()
            opt.step()

            print('ep {}, batch {}, cost {}'.format(ep, i, cost.item()))
        if ep % 5 == 0:
            im = ut.collate_images(xhat, 64)
            opts = {'title': 'reconstructions'}
            vis.heatmap(im, opts=opts, win='xhat')

            im = ut.collate_images(tar, 64)
            opts = {'title': 'targets'}
            vis.heatmap(im, opts=opts, win='x')

    torch.save(enc.state_dict(), '/mnist_encoder.pt')        

all_hhats = get_embeddings(enc, dataset_loader)

# fit the GMM on the latent embeddings 
all_scores_ganmog = []
for Kcomps in range(50, 51, 5):
    print('Number of GMM components {}'.format(Kcomps))

    dec.GMM = mix.GaussianMixture(n_components=Kcomps, verbose=1, n_init=1, max_iter=200, covariance_type='full', warm_start=True)
    dec.GMM.fit(all_hhats.data.cpu().numpy())

    scores_ganmog, _ = get_scores(test_loader, dec, arguments.cuda, 
                               num_samples=num_samples,
                               task='mnist', base_dist='mog_skt')
    all_scores_ganmog.append({'Kcomps' : Kcomps,
                              'scores_ganmog' : scores_ganmog})
    
