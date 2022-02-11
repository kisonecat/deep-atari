import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
#from torchsummary import summary
import torchvision
from PIL import Image

from ale_py import ALEInterface
from random import randrange

#print(ale.getScreenRGB())
#print(ale.getScreenDims())
#print(len(ale.getRAM()))
#print(dir(ale))

#for episode in range(10):
#    while not ale.game_over():
#        a = legal_actions[randrange(len(legal_actions))]
#        ale.act(a)
#    print("resetting",episode)
#    ale.reset_game()

class AtariDataset(IterableDataset):
    def __init__(self):
        self.ale = ALEInterface()
        self.ale.loadROM('river-raid.bin')
        self.legal_actions=self.ale.getLegalActionSet()

        self.ale.reset_game()
        for _ in range(2):
            self.ale.setRAM( randrange(128), randrange(256) )
        
    def __iter__(self):
        return self

    def __next__(self):
        memory = self.ale.getRAM()
        self.ale.act( self.legal_actions[randrange(len(self.legal_actions))] )
        image = self.ale.getScreenRGB()

        if self.ale.game_over():
            self.ale.reset_game()
            for _ in range(2):
                self.ale.setRAM( randrange(128), randrange(256) )
            
        memory = np.reshape( np.unpackbits(memory), (128*8) )
        memory = memory.astype(float)

        image = Image.fromarray(image)
        image = torchvision.transforms.Resize( (128,128) ).forward( image )
        #image.save('tester' + str(randrange(128)) + '.png')
        image = np.array(image) / 255.0
        image = image.transpose( (2, 0, 1) )
        return memory, image

# custom weights initialization called on generator and discriminator
def init_weights(net, init_type='normal', scaling=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func) 

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        model = []

        self.fc = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 16, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(16, 16, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
        )    

        self.model = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ) 
        
        return

    def forward(self, input):
        input = torch.reshape( input, (-1, 1, 8*128) )
        input = self.fc( input )
        width = 16 
        input = torch.reshape( input, (-1, int(16384 / width / width), width, width) )
        input = self.model( input )
        return input 

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(32*32, 32*8),
            nn.ReLU(True),
            nn.Linear(32*8, 32*8),
            nn.ReLU(True),
            nn.Linear(32*8, 4*128*128),
            nn.ReLU(True),
        )    
        
        self.combiner = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(True),
            nn.Conv1d(8, 16, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=9, stride=1, padding=4, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
        )    

        kw = 4
        padw = 1
        input_nc = 5 
        ndf = 16
        n_layers = 4
        norm_layer = nn.BatchNorm2d
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), nn.Sigmoid()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input, output):
        input = torch.reshape( input, (-1, 1, 8*128) )
        input = self.combiner( input )
        
        x = torch.reshape( input, (-1, 2, 128, 128) )
        together = torch.cat((x, output), dim=1)
        return self.model(together)

batch_size = 128 

dataset = AtariDataset()
#xs = dataset[0]
#print(xs[0].shape)
#print(xs[1].shape)
#(32, 32)
#(210, 160)

train_dl = DataLoader(dataset, batch_size)

if torch.cuda.is_available():
    print('cuda is available')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device count',torch.cuda.device_count())

generator = Generator().to(device).float()
init_weights(generator, 'normal', scaling=0.02)
adversarial_loss = nn.BCELoss() 
l1_loss = nn.L1Loss()

#for x in train_dl:
#    print(x)
#    print(summary(generator,x[0].shape))

discriminator = Discriminator().to(device)

def generator_loss(generated_image, target_img, G, real_target):
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (100 * l1_l)
    return gen_total_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

lr = 2e-4
b1 = 0.5
b2 = 0.999
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

start_epoch = 47 
print('loading model from epoch',start_epoch)
generator.load_state_dict(torch.load('generator%03d.pt' % start_epoch))
generator.eval()
discriminator.load_state_dict(torch.load('discriminator%03d.pt' % start_epoch))
discriminator.eval()

num_epochs = 200
D_loss_plot, G_loss_plot = [], []
for epoch in range(start_epoch, start_epoch + num_epochs): 
    print('epoch',epoch)
    D_loss_list, G_loss_list = [], []

    num_batches = 64 
    batch_count = 0
    for (input_img, target_img) in train_dl:
        D_optimizer.zero_grad()
        input_img = input_img.float().to(device)
        target_img = target_img.float().to(device)

        # ground truth labels real and fake
        count = 6
        real_target = Variable(torch.ones(input_img.size(0), 1, count, count).to(device))
        fake_target = Variable(torch.zeros(input_img.size(0), 1, count, count).to(device))
        
        # generator forward pass
        generated_image = generator(input_img)
        # train discriminator with fake/generated images
        D_fake = discriminator(input_img.detach(), generated_image.detach())
        
        D_fake_loss   =  discriminator_loss(D_fake, fake_target)
        
        # train discriminator with real images
        #disc_inp_real = torch.cat((input_img, target_img), 1)
        #D_real = discriminator(disc_inp_real)
        D_real = discriminator( input_img, target_img )
        D_real_loss = discriminator_loss(D_real,  real_target)

        # average discriminator loss
        D_total_loss = (D_real_loss + D_fake_loss) / 2
        D_loss_list.append(D_total_loss)
        # compute gradients and run optimizer step
        D_total_loss.backward()
        D_optimizer.step()
        
        # Train generator with real labels
        G_optimizer.zero_grad()
        G = discriminator(input_img, generated_image)
        G_loss = generator_loss(generated_image, target_img, G, real_target)

        batch_count = batch_count + 1
        print('epoch',epoch,'batch',batch_count,'of',num_batches,G_loss)   
        print('D_fake_loss', D_fake_loss)
        print('D_real_loss', D_real_loss)
        
        G_loss_list.append(G_loss)
        # compute gradients and run optimizer step
        G_loss.backward()
        G_optimizer.step()

        if batch_count >= num_batches:
            compare = Image.new('RGB', (256, 256))

            x = generated_image.cpu().detach().numpy()[0]
            x = x.transpose((1, 2, 0))
            x = (x * 255).astype(np.uint8)
            x = Image.fromarray(x)

            y = target_img.cpu().detach().numpy()[0]
            y = y.transpose((1, 2, 0))
            y = (y * 255).astype(np.uint8)
            y = Image.fromarray(y)

            compare.paste(y, (0,0))
            compare.paste(x, (128,0))

            x = generated_image.cpu().detach().numpy()[batch_size-1]
            x = x.transpose((1, 2, 0))
            x = (x * 255).astype(np.uint8)
            x = Image.fromarray(x)

            y = target_img.cpu().detach().numpy()[batch_size-1]
            y = y.transpose((1, 2, 0))
            y = (y * 255).astype(np.uint8)
            y = Image.fromarray(y)

            compare.paste(y, (0,128))
            compare.paste(x, (128,128))

            compare.save('compare%03d.png' % epoch )

            break

    print('saving model...')
    torch.save(generator.state_dict(), 'generator%03d.pt' % epoch)
    torch.save(discriminator.state_dict(), 'discriminator%03d.pt' % epoch)
