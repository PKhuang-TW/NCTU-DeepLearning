import torch
import torch.nn as nn

class Shared(nn.Module):
    ''' front end part of discriminator and Q'''

    def __init__(self):
        super(Shared, self).__init__()

        self.main = nn.Sequential(            
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        output = self.main(x)
        return output


class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output


class Q(nn.Module):

    def __init__(self):
        super(Q, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=8192, out_features=100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=10, bias=True)
        )
        self.conv_disc = nn.Conv2d(10, 10, 1)
        self.conv_mu = nn.Conv2d(10, 2, 1)
        self.conv_var = nn.Conv2d(10, 2, 1)
        
    def forward(self, x):

        x = x.view(-1, 8192)
        y = self.main(x)
        y = y.view(-1, 10, 1, 1)

        disc_logits = self.conv_disc(y).squeeze()

        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()

        return disc_logits, mu, var


class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(74, 512, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.main(x)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)