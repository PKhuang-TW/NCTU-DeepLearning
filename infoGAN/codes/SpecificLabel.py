import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

import numpy as np

from model import *


fe = Shared().cuda()
d = D().cuda()
q = Q().cuda()
g = G().cuda()

bestModelPath = '../models/0523_0431/'

fe.load_state_dict(torch.load(bestModelPath + 'FE_weight.pt'))
d.load_state_dict(torch.load(bestModelPath + 'D_weight.pt'))
q.load_state_dict(torch.load(bestModelPath + 'Q_weight.pt'))
g.load_state_dict(torch.load(bestModelPath + 'G_weight.pt'))


# fixed random variables
c = np.linspace(-1, 1, 10).reshape(1, -1)
c = np.repeat(c, 10).reshape(-1, 1)

zeros = np.zeros(c.shape)

c1 = np.hstack([c, zeros])
c2 = np.hstack([zeros, c])


dis_c = torch.FloatTensor(100, 10).cuda()
con_c = torch.FloatTensor(100, 2).cuda()
noise = torch.FloatTensor(100, 62).cuda()
dis_c = Variable(dis_c)
con_c = Variable(con_c)
noise = Variable(noise)



for label in range(10):
    one_hot = np.zeros((100, 10))
    one_hot[:, label] = 1

    dis_c = dis_c.data.copy_(torch.Tensor(one_hot))

    fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)
    noise = noise.data.copy_(fix_noise)

    con_c.data.copy_(torch.from_numpy(c1))
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
    x = g(z)
    save_image(x.data, '../results/SpecificLabel/{}_c1.png'.format(label), nrow=10)

    con_c.data.copy_(torch.from_numpy(c2))
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
    x = g(z)
    save_image(x.data, '../results/SpecificLabel/{}_c2.png'.format(label), nrow=10)