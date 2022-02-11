import torch
import torch.nn as nn

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
