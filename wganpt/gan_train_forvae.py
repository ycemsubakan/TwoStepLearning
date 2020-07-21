from timeit import default_timer as timer
import os, sys
sys.path.append(os.getcwd())

path = '..'
sys.path.insert(0, path)
from algorithms_v2 import VAE, get_embeddings
import utils as ut

import time
import functools
import argparse

import numpy as np
#import sklearn.datasets

#import models.dcgan as dcgan

import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
import torch.utils.data as data_utils

import torch.nn.init as init
import pdb
import torch.nn.functional as F
import visdom



class vaegan_g(nn.Module):
    def __init__(self, Kz=40, K=1024):
        super(vaegan_g, self).__init__()
        self.net = nn.Sequential(
                   nn.Linear(Kz, K),
                   nn.ReLU(),
                   nn.Linear(K, K), 
                   nn.ReLU(),
                   nn.Linear(K, K),
                   nn.ReLU(), 
                   nn.Linear(K, K),
                   nn.ReLU(),
                   nn.Linear(K, 2*Kz),
                   )

    def forward(self, z):
        Kz = z.size(1)

        zp = self.net.forward(z)
        gt = F.sigmoid(zp[:, :Kz])
        dz = zp[:, Kz:]

        return (1-gt)*z + gt*dz

class vaegan_d(nn.Module):
    def __init__(self, Kz=40, K=1024):
        super(vaegan_d, self).__init__()
        self.net = nn.Sequential(
                   nn.Linear(Kz, K),
                   nn.ReLU(),
                   nn.Linear(K, K), 
                   nn.ReLU(),
                   nn.Linear(K, K),
                   nn.ReLU(), 
                   nn.Linear(K, K),
                   nn.ReLU(),
                   nn.Linear(K, 1),
                   )

    def forward(self, z):

        zp = (self.net.forward(z))

        return zp



# this code is based on https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py
def train_wgangp(dataloader):
    RESTORE_MODE = False # if True, it will load saved model from OUT_PATH and continue to train
    START_ITER = 0 # starting iteration 
    OUTPUT_PATH = 'vaeganprior/' # output path where result (.e.g drawing images, cost, chart) will be stored
    MODE = 'wgan-gp' # dcgan, wgan
    DIM = 20 # Model dimensionality
    CRITIC_ITERS = 5 # How many iterations to train the critic for
    GENER_ITERS = 1
    N_GPUS = 2 # Number of GPUs
    BATCH_SIZE = 3000 # Batch size. Must be a multiple of N_GPUS
    END_ITER = 100000 # How many iterations to train for
    LAMBDA = 10 # Gradient penalty lambda hyperparameter
    OUTPUT_DIM = DIM # Number of pixels in each image

    fixed_noise = gen_rand_noise(BATCH_SIZE) 

    aG = vaegan_g(Kz=DIM)
    aD = vaegan_d(Kz=DIM)
        
    LR = 2e-5
    optimizer_g = torch.optim.Adam(aG.parameters(), lr=LR, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=LR, betas=(0,0.9))
    one = torch.FloatTensor([1])
    mone = one * -1
    aG = aG.to(device)
    aD = aD.to(device)
    one = one.to(device)
    mone = mone.to(device)

    dataiter = iter(dataloader)
    for iteration in range(START_ITER, END_ITER):
        start_time = time.time()
        print("Iter: " + str(iteration))
        start = timer()
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        gen_cost = None
        for i in range(GENER_ITERS):
            print("Generator iters: " + str(i))
            aG.zero_grad()
            noise = gen_rand_noise(BATCH_SIZE)
            noise.requires_grad_(True)
            fake_data = aG(noise)
            gen_cost = aD(fake_data)
            gen_cost = gen_cost.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost
        
        optimizer_g.step()
        end = timer()
        print(f'---train G elapsed time: {end - start}')
        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G
        for i in range(CRITIC_ITERS):
            print("Critic iter: " + str(i))
            
            start = timer()
            aD.zero_grad()

            # gen fake data and load real data
            noise = gen_rand_noise(BATCH_SIZE)
            with torch.no_grad():
                noisev = noise  # totally freeze G, training D
            fake_data = aG(noisev).detach()
            end = timer(); print(f'---gen G elapsed time: {end-start}')
            start = timer()
            batch = next(dataiter, None)
            if batch is None:
                dataiter = iter(dataloader)
                batch = dataiter.next()
            batch = batch[0] #batch[1] contains labels
            real_data = batch.to(device) #TODO: modify load_data for each loading
            
            real_data = real_data.view(real_data.size(0), -1)

            end = timer(); print(f'---load real imgs elapsed time: {end-start}')
            start = timer()

            # train with real data
            disc_real = aD(real_data)
            disc_real = disc_real.mean()

            # train with fake data
            disc_fake = aD(fake_data)
            disc_fake = disc_fake.mean()

            #showMemoryUsage(0)
            # train with interpolates data

            gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data, BATCH_SIZE, LAMBDA)
            #showMemoryUsage(0)

            # final disc cost
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake  - disc_real
            optimizer_d.step()
            #------------------VISUALIZATION----------
            
            end = timer(); print(f'---train D elapsed time: {end-start}')
        #---------------VISUALIZATION---------------------
        if iteration % 50 == 0:
            gen_images = generate_image(mdl, aG, fixed_noise)
            vis.images(gen_images.reshape(-1, 1, 28, 28), win='vaeganprior_genims')

	#----------------------Save model----------------------
            torch.save(aG, OUTPUT_PATH + "generator.pt")
            torch.save(aD, OUTPUT_PATH + "discriminator.pt")

def weights_init(m):
    if isinstance(m, MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                init.kaiming_uniform_(m.conv.weight)
            else:
                init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE, LAMBDA):
    alpha = torch.rand(BATCH_SIZE)
    #alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()

    DIM1 = 28
    DIM2 = 28
    #alpha = alpha.view(BATCH_SIZE, 3, DIM1, DIM2)
    alpha = alpha.view(-1, 1)
    alpha = alpha.to(device)
    
    #fake_data = fake_data.view(BATCH_SIZE, 3, DIM1, DIM2)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def generate_image(vaemdl, netG, noise=None):
    if noise is None:
        noise = gen_rand_noise(100)

    with torch.no_grad():
    	noisev = noise 
    noise_samples = netG(noisev)

    samples = vaemdl.decode(noise_samples)
    samples = samples.view(-1, 1, 28, 28)
    return samples

def gen_rand_noise(BATCH_SIZE, K=20):
    noise = torch.randn(BATCH_SIZE, K)
    noise = noise.to(device)

    return noise



if __name__ == '__main__': 
    vis = visdom.Visdom(port=5800, server='http://', env='')
    assert vis.check_connection()

    class args():
        def __init__(self):
            self.data = 'mnist'
            self.cuda = True
    arguments = args()

    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'

    modelpath = '../models/VAE_{}_K_{}.t'.format('mnist', '[20, 600]')

    # load the VAE
    mdl = VAE(784, 784, [20, 600], M=28) 

    if os.path.exists(path):
        mdl.load_state_dict(torch.load(modelpath))
    mdl = mdl.to(device)

    
    # fit the wgangp
    generator_path = 'vaeganprior/generator.pt'
    if os.path.exists(generator_path):
        
        gen_images_vae, seed = mdl.generate_data(100) 
        torchvision.utils.save_image(gen_images_vae.reshape(-1, 1, 28, 28), 'generations_vae_mnist.png') 
        #vis.images(gen_images_vae.reshape(-1, 1, 28, 28), win='vae_genims')

        aG = torch.load(generator_path)

        gen_images_vaeganprior = generate_image(mdl, aG, seed)
        torchvision.utils.save_image(gen_images_vaeganprior, 'generations_vaeganprior_mnist.png') 
        #vis.images(gen_images_vaeganprior.reshape(-1, 1, 28, 28), win='vaeganp_genims')
    else:
        # get the embeddings
        train_loader, _ = ut.get_loaders(3000, c=0, train_shuffle=True, arguments=arguments)
        hhat = get_embeddings(mdl, train_loader)

        hhat = data_utils.TensorDataset(hhat)
        data_loader = data_utils.DataLoader(hhat, batch_size=3000, shuffle=True)

        # train the gan
        train_wgangp(data_loader)
        




