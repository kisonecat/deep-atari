import torch
import torch.nn as nn

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

