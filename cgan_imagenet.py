import os, sys
sys.path.append(os.getcwd())

import time
#import tflib as lib
#import tflib.save_images
#import tflib.mnist
#import tflib.cifar10
#import tflib.plot
#import tflib.inception_score

import numpy as np


import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets

from pytorch_utils import get_model, hook_get_shapes, hook_get_acts, save_checkpoint

from networks import define_G, define_D

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = '/data/datasets/ILSVRC2012'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

Z_DIM = 100 
HIDDEN_DIM = 64 # This overfits substantially; you're probably better off with 64
LAMBDA = 10 # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 32 # Batch size
NUM_EXAMPLES = 5
ITERS = 200000 # How many generator iterations to train for
D_NAME = 'dcgan_nobn_basic'
G_NAME = 'resnet_6blocks'
PRINT_ITER = 5 # How often to print to screen
SAVE_ITER = 50
ARCH = 'alexnet'
BLOB = 'features.10'
CHECKPOINT = None
SIZE = 128
RESULTS_DIR = 'cgan_imagenet_%s_%s_%d_G_%s_D_%s_zdim_%d_hdim_%d' % (ARCH, BLOB, SIZE, G_NAME, D_NAME, Z_DIM, HIDDEN_DIM)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0

# Dataset iterator
preprocess = torchvision.transforms.Compose([
                               torchvision.transforms.Resize(SIZE),
                               torchvision.transforms.CenterCrop(SIZE),
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

def inf_train_gen():
    dset = datasets.ImageFolder(os.path.join(DATA_DIR, 'images', 'train'), transform=preprocess)
    loader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, 
                                         num_workers=4, pin_memory=use_cuda)
    while True:
        for (x, _) in loader:
            yield x

gen = inf_train_gen()
dev_loader = DataLoader(datasets.ImageFolder(os.path.join(DATA_DIR, 'images', 'val'), transform=preprocess),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=use_cuda)

x = gen.next()
x = autograd.Variable(x.cuda() if use_cuda else x)
model = get_model(arch=ARCH, dataset='imagenet', pretrained=True, adaptive_pool=True,
                  checkpoint_path=CHECKPOINT, cuda=use_cuda)

y_shape = hook_get_shapes(model, [BLOB], x)[0]

netD = define_D(y_shape[1]+x.shape[1], x.shape[-1], HIDDEN_DIM, D_NAME, norm='instance')
netG = define_G(y_shape[1]+Z_DIM, x.shape[1], x.shape[-1], HIDDEN_DIM, G_NAME, norm='instance')
print netG
print netD

if use_cuda:
    netD = netD.cuda(gpu)
    netG = netG.cuda(gpu)

one = torch.FloatTensor([1])
mone = one * -1
if use_cuda:
    one = one.cuda(gpu)
    mone = mone.cuda(gpu)

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))

def calc_gradient_penalty(netD, real_data, fake_data, y):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 3, SIZE, SIZE)
    alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, y.detach())

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

# For generating samples
fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
if use_cuda:
    fixed_noise = fixed_noise.cuda(gpu)
noisev = autograd.Variable(fixed_noise, volatile=True)
def generate_image(frame, netG, y, real_data):
    samples = netG(noisev, y.detach())
    samples = samples.view(-1, 3, SIZE, SIZE)
    #samples = samples.mul(0.5).add(0.5)
    #samples = samples.cpu().data.numpy()
    filename = os.path.join(RESULTS_DIR, str(frame+1) + ".jpg")
    torchvision.utils.save_image(torch.cat((samples.data.cpu(), real_data.data.cpu()), dim=0), filename, normalize=True)

    #lib.save_images.save_images(samples, './tmp/cifar10/samples_{}.jpg'.format(frame))

def generate_samples(frame, netG, real_imgs, model):
    num_examples = real_imgs.shape[0]
    x = torch.FloatTensor(num_examples ** 2, real_imgs.shape[1], real_imgs.shape[2], real_imgs.shape[3])
    x = autograd.Variable(x.cuda() if use_cuda else x)

    noise = torch.FloatTensor(num_examples, Z_DIM, 1, 1).normal_(0, 1)
    z = torch.FloatTensor(num_examples ** 2, Z_DIM, 1, 1)
    z = autograd.Variable(z.cuda() if use_cuda else z)

    for i in range(num_examples):
        x.data[i] = real_imgs[i]
        for j in range(1, num_examples):
            z.data[i*num_examples + j] = noise[i]
            x.data[j*num_examples + i] = real_imgs[i]

    y = hook_get_acts(model, [BLOB], x)[0]
    gen_x = netG(z[num_examples:], y[num_examples:])
    x[num_examples:] = gen_x
    filename = os.path.join(RESULTS_DIR, 'vary_z_%d.jpg' % (frame+1))
    torchvision.utils.save_image(x.data.cpu(), filename, nrow=num_examples, normalize=True)


D_costs = []
G_costs = []
W_diffs = []

for iteration in xrange(ITERS):
    start_time = time.time()
    ############################
    # (1) Update D network
    ###########################
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for i in xrange(CRITIC_ITERS):
        #_data = gen.next()
        real_data = gen.next()
        if real_data.shape[0] != BATCH_SIZE:
            real_data = gen.next()
        netD.zero_grad()

        # train with real
        #_data = _data.reshape(BATCH_SIZE, 3, 32, 32).transpose(0, 2, 3, 1)
        #real_data = torch.stack([preprocess(item) for item in _data])

        if use_cuda:
            real_data = real_data.cuda(gpu)
        real_data_v = autograd.Variable(real_data)

        y = hook_get_acts(model, [BLOB], real_data_v)[0]

        # import torchvision
        # filename = os.path.join("test_train_data", str(iteration) + str(i) + ".jpg")
        # torchvision.utils.save_image(real_data, filename)

        D_real = netD(real_data_v, y.detach())
        D_real = D_real.mean()
        D_real.backward(mone)

        # train with fake
        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
        if use_cuda:
            noise = noise.cuda(gpu)
        noisev = autograd.Variable(noise, volatile=True)  # totally freeze netG
        fake = autograd.Variable(netG(noisev, y.detach()).data)
        inputv = fake
        D_fake = netD(inputv, y.detach())
        D_fake = D_fake.mean()
        D_fake.backward(one)

        # train with gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data, y)
        gradient_penalty.backward()

        # print "gradien_penalty: ", gradient_penalty

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake
        optimizerD.step()
    ############################
    # (2) Update G network
    ###########################
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation
    netG.zero_grad()

    noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1)
    if use_cuda:
        noise = noise.cuda(gpu)
    noisev = autograd.Variable(noise)
    fake = netG(noisev, y.detach())
    G = netD(fake, y.detach())
    G = G.mean()
    G.backward(mone)
    G_cost = -G
    optimizerG.step()

    D_costs.append(D_cost.data.cpu().numpy()[0])
    G_costs.append(G_cost.data.cpu().numpy()[0])
    W_diffs.append(Wasserstein_D.data.cpu().numpy()[0])

    if (iteration) % PRINT_ITER == 0:
        print('Train [%d/%d] D: %.4f G: %.4f W: %.4f T: %.4f' % (iteration+1, ITERS, 
              D_cost.data.cpu().numpy()[0], G_cost.data.cpu().numpy()[0],
              Wasserstein_D.data.cpu().numpy()[0], time.time() - start_time))

    # Write logs and save samples
    #lib.plot.plot('./tmp/cifar10/train disc cost', D_cost.cpu().data.numpy())
    #lib.plot.plot('./tmp/cifar10/time', time.time() - start_time)
    #lib.plot.plot('./tmp/cifar10/train gen cost', G_cost.cpu().data.numpy())
    #lib.plot.plot('./tmp/cifar10/wasserstein distance', Wasserstein_D.cpu().data.numpy())

    # Calculate inception score every 1K iters
    #if False and iteration % 1000 == 999:
        #inception_score = get_inception_score(netG)
        #lib.plot.plot('./tmp/cifar10/inception score', inception_score[0])

    # Calculate dev loss and generate samples every 100 iters
    if (iteration) % SAVE_ITER == 0:
        #dev_disc_costs = []
        #for imgs, _ in dev_loader:
        #    # imgs = preprocess(images)
        #    if use_cuda:
        #        imgs = imgs.cuda(gpu)
        #    imgs_v = autograd.Variable(imgs, volatile=True)
        #
        #    y = hook_get_acts(model, [BLOB], imgs_v)[0]
        #    D = netD(imgs_v, y)
        #    _dev_disc_cost = -D.mean().cpu().data.numpy()
        #    dev_disc_costs.append(_dev_disc_cost)
        #print('Test [%d/%d] Avg D: %.4f' % (iteration+1, ITERS, np.mean(dev_disc_costs)))
        #lib.plot.plot('./tmp/cifar10/dev disc cost', np.mean(dev_disc_costs))

        imgs, _ = next(iter(dev_loader))
        imgs_v = autograd.Variable(imgs.cuda() if use_cuda else imgs, volatile=True)
        y = hook_get_acts(model, [BLOB], imgs_v)[0]
        generate_image(iteration, netG, y, imgs_v)
        generate_samples(iteration, netG, imgs_v[:NUM_EXAMPLES].data, model)

        save_checkpoint({
                         'state_dict': netG.state_dict(), 
                         'optimizer': optimizerG.state_dict(),
                         'iteration': iteration+1,
                         'G_costs': G_costs,
                         'D_costs': D_costs,
                         'W_diffs': W_diffs,
                         }, 
                         '%s/G_checkpoint.pth.tar' % RESULTS_DIR)
        save_checkpoint({
                         'state_dict': netD.state_dict(),
                         'optimizer': optimizerD.state_dict(),
                         'iteration': iteration+1,
                         },
                         '%s/D_checkpoint.pth.tar' % RESULTS_DIR)

    # Save logs every 100 iters
    #if (iteration < 5) or (iteration % 100 == 99):
    #    lib.plot.flush()
    #lib.plot.tick()
