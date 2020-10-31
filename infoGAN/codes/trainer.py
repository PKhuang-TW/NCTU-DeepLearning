import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

dataset = dset.MNIST(root='../dataset', download=True,
					 transform=transforms.Compose([
						 transforms.Resize(64),
						 transforms.ToTensor(),
					 ]))
dataloader = DataLoader(dataset, batch_size=100, shuffle=True)


class log_gaussian:

	def __call__(self, x, mu, var):

		logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - (x-mu).pow(2).div(var.mul(2.0)+1e-6)

		return logli.sum(1).mean().mul(-1)

	
class Trainer:

	def __init__(self, G, shared, D, Q, EPOCH):

		self.G = G
		self.shared = shared
		self.D = D
		self.Q = Q
		
		self.EPOCH = EPOCH
		self.batch_size = 100
		
	def _noise_sample(self, dis_c, con_c, noise, bs):

		idx = np.random.randint(10, size=bs)
		c = np.zeros((bs, 10))
		c[range(bs), idx] = 1.0  # Random one hot code

		dis_c.data.copy_(torch.Tensor(c))
		con_c.data.uniform_(-1.0, 1.0)
		noise.data.uniform_(-1.0, 1.0)
		z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)

		return z, idx
		
	def train(self):

		DlossList = []
		QlossList = []
		GlossList = []
		
		realProb_before = []
		realProb_after = []
		fakeProb_before = []
		fakeProb_after = []

		real_x = torch.FloatTensor(self.batch_size, 1, 64, 64).cuda()
		label = torch.FloatTensor(self.batch_size, 1).cuda()
		dis_c = torch.FloatTensor(self.batch_size, 10).cuda()
		con_c = torch.FloatTensor(self.batch_size, 2).cuda()
		noise = torch.FloatTensor(self.batch_size, 62).cuda()

		real_x = Variable(real_x)
		label = Variable(label, requires_grad=False)
		dis_c = Variable(dis_c)
		con_c = Variable(con_c)
		noise = Variable(noise)

		criterionD = nn.BCELoss().cuda()
		criterionQ_dis = nn.CrossEntropyLoss().cuda()
		criterionQ_con = log_gaussian()

		optimD = optim.Adam([{'params': self.shared.parameters()}, {
							'params': self.D.parameters()}], lr=0.0002, betas=(0.5, 0.99))
		optimG = optim.Adam([{'params': self.G.parameters()}, {
							'params': self.Q.parameters()}], lr=0.001, betas=(0.5, 0.99))

		# fixed random variables
		c = np.linspace(-1, 1, 10).reshape(1, -1)
		c = np.repeat(c, 10).reshape(-1, 1)

		zeros = np.zeros(c.shape)

		c1 = np.hstack([c, zeros])
		c2 = np.hstack([zeros, c])

		# idx = np.arange(10).repeat(10)
		# one_hot = np.zeros((100, 10))
		# one_hot[range(100), idx] = 1
		one_hot = np.zeros((100, 10))
		for i in range(100):
			one_hot[i, i%10] = 1
		fix_noise = torch.Tensor(100, 62).uniform_(-1, 1)

		for epoch in tqdm(range(self.EPOCH)):
			for num_iters, batch_data in enumerate(dataloader, 0):

				x, _ = batch_data
				bs = x.size(0)  # data size

				# Initial size of data, label, noise and r.v. c
				real_x.data.resize_(x.size())
				label.data.resize_(bs, 1)
				dis_c.data.resize_(bs, 10)
				con_c.data.resize_(bs, 2)
				noise.data.resize_(bs, 62)

				######################
				##### Learning D #####
				######################

				### real part ###
				optimD.zero_grad()
				real_x.data.copy_(x)
				shared_out1 = self.shared(real_x)
				probs_real = self.D(shared_out1)
				realProb_before.append(probs_real.mean().data.cpu().numpy()) # Append Real Prob before updating D
				label.data.fill_(1)  # Set initial label all as 1
				loss_real = criterionD(probs_real, label)
				loss_real.backward()

				### fake part###
				z, idx = self._noise_sample(dis_c, con_c, noise, bs)
				fake_x = self.G(z)
				shared_out2 = self.shared(fake_x.detach())
				probs_fake = self.D(shared_out2)
				fakeProb_before.append(probs_fake.mean().data.cpu().numpy()) # Append Fake Prob before updating G
				label.data.fill_(0)  # Set initial label all as 0
				loss_fake = criterionD(probs_fake, label)
				loss_fake.backward()

				D_loss = loss_real + loss_fake
				DlossList.append(D_loss.data.cpu().numpy())  # Append D Loss

				optimD.step()
				
				### Append Real Prob after updating D ###
				shared_out1 = self.shared(real_x)
				probs_real = self.D(shared_out1)
				realProb_after.append(probs_real.mean().data.cpu().numpy()) # Append Real Prob after updating D

				############################
				##### Learning G and Q #####
				############################

				### G and Q part ###
				optimG.zero_grad()

				shared_out = self.shared(fake_x)
				probs_fake = self.D(shared_out)
				label.data.fill_(1.0)
				reconstruct_loss = criterionD(probs_fake, label)

				q_logits, q_mu, q_var = self.Q(shared_out)
				class_ = torch.LongTensor(idx).cuda()
				target = Variable(class_)
				dis_loss = criterionQ_dis(q_logits, target)
				con_loss = criterionQ_con(con_c, q_mu, q_var)*0.1
				QlossList.append((dis_loss + con_loss).data.cpu().numpy())  # Append Q Loss

				G_loss = reconstruct_loss + dis_loss + con_loss
				GlossList.append(G_loss.data.cpu().numpy()) # Append G Loss
				G_loss.backward()
				optimG.step()
				
				### Append Fake Prob after updating G ###
				fake_x = self.G(z)
				shared_out2 = self.shared(fake_x.detach())
				probs_fake = self.D(shared_out2)
				fakeProb_after.append(probs_fake.mean().data.cpu().numpy())   # Append Fake Prob after updating G
				
				if num_iters % 100 == 0:

#                     print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
#                         epoch, num_iters, D_loss.data.cpu().numpy(),
#                         G_loss.data.cpu().numpy())
#                     )

					noise.data.copy_(fix_noise)
					dis_c.data.copy_(torch.Tensor(one_hot))

					con_c.data.copy_(torch.from_numpy(c1))
					z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
					x_save = self.G(z)
					save_image(x_save.data, '../results/c1/{}.png'.format(str(epoch*len(dataloader) + num_iters)), nrow=10)

					con_c.data.copy_(torch.from_numpy(c2))
					z = torch.cat([noise, dis_c, con_c], 1).view(-1, 74, 1, 1)
					x_save = self.G(z)
					save_image(x_save.data, '../results/c2/{}.png'.format(str(epoch*len(dataloader) + num_iters)), nrow=10)
		
		currTime = time.strftime("%m%d_%H%M", time.localtime())
		modelPath = '../models/'+currTime+'/'
		os.makedirs(modelPath)
		torch.save(self.shared.state_dict(), modelPath+'shared_weight.pt')
		torch.save(self.D.state_dict(), modelPath+'D_weight.pt')
		torch.save(self.Q.state_dict(), modelPath+'Q_weight.pt')
		torch.save(self.G.state_dict(), modelPath+'G_weight.pt')
		
		plot_lossNprob(DlossList, QlossList, GlossList, realProb_before, realProb_after, fakeProb_before, fakeProb_after, modelPath, self.EPOCH)
		

def plot_lossNprob(DlossList, QlossList, GlossList, realProb_before, realProb_after, fakeProb_before, fakeProb_after, modelPath, EPOCH):
	new_DlossList = []
	new_GlossList = []
	new_QlossList = []
	new_realProb_before = []
	new_realProb_after = []
	new_fakeProb_before = []
	new_fakeProb_after  = []
	
	ptr = 0
	for idx in range(len(DlossList)):
		if ((idx+1)%100 == 0):
			new_DlossList.append(sum(DlossList[ptr:ptr+100])/100)
			new_GlossList.append(sum(GlossList[ptr:ptr+100])/100)
			new_QlossList.append(sum(QlossList[ptr:ptr+100])/100)
			new_realProb_before.append(sum(realProb_before[ptr:ptr+100])/100)
			new_realProb_after.append(sum(realProb_after[ptr:ptr+100])/100)
			new_fakeProb_before.append(sum(fakeProb_before[ptr:ptr+100])/100)
			new_fakeProb_after.append(sum(fakeProb_after[ptr:ptr+100])/100)
			ptr = idx
	
	fig = plt.figure(figsize=(20,6))
	xs = range(int(EPOCH*len(dataloader)/100))
	plt.plot(xs, new_DlossList, label='Dloss')
	plt.plot(xs, new_GlossList, label='Gloss')
	plt.plot(xs, new_QlossList, label='Qloss')
	plt.legend(loc='best'), plt.savefig(modelPath+"Loss.png"), plt.show()
	
	fig = plt.figure(figsize=(20,6))
	xs = range(int(EPOCH*len(dataloader)/100))
	plt.plot(xs, new_realProb_before, label='Real Prob (Before)')
	plt.plot(xs, new_realProb_after, label='Real Prob (After)')
	plt.plot(xs, new_fakeProb_before, label='Fake Prob (Before)')
	plt.plot(xs, new_fakeProb_after, label='Fake Prob (After)')
	plt.legend(loc='best'), plt.savefig(modelPath+"Prob.png"), plt.show()