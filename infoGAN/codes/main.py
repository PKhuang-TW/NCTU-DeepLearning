from model import *
from trainer import *

EPOCH = 1

shared = Shared()
d = D()
q = Q()
g = G()

for i in [shared, d, q, g]:
    i.cuda()
    i.apply(weights_init)

trainer = Trainer(g, shared, d, q, EPOCH)
trainer.train()