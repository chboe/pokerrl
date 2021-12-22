import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
# import re
import time

Tensor = torch.FloatTensor

nd = torch.distributions.normal.Normal(Tensor([0.5]), Tensor([0.25]))
# nd = torch.normal(mean=Tensor([0.5]), std=Tensor([0.25]))

print(nd.log_prob(Tensor([0.7])))




# MODEL_TO_LOAD = 'Agents/NFSP_Model/id=1_steps=2000'
# steps = int(re.findall(r'steps=(\d*)', MODEL_TO_LOAD)[-1])
# print(steps)


# class Network(nn.Module):
#     def __init__(self):
#         nn.Module.__init__(self)
#         self.l1 = nn.Linear(288, 64)
#         self.l2 = nn.Linear(64, 128)
#         self.l3 = nn.Linear(128, 128)
#         self.l4 = nn.Linear(128, 64)
#         self.l5 = nn.Linear(64, 3)

#     def forward(self, x):
#         x = F.relu(self.l1(x))
#         print(x)
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         x = F.relu(self.l4(x))
#         x = self.l5(x)
#         return x


# batch = torch.ones((50,3,4,5))

# g = batch.flatten(1)
# print(g.shape)

# before = time.time()

# print(time.time() - before)


# linearnn = Network()
# rnn = nn.RNN(6, 80, batch_first=True)

# linearOpt = optim.Adam(linearnn.parameters(), 0.1)
# rnnOpt = optim.SGD(rnn.parameters(), 0.1)


# input_example = torch.ones((1, 10, 6))

# _, h_n = rnn(input_example)

# x = h_n.flatten()

# cards = torch.ones((208), requires_grad=True)
# input_to_linear = torch.cat((x, cards))


# y = linearnn(input_to_linear)


# print("BEFORE")
# for param in rnn.parameters():
#     print(param[0])
#     break

# rnnOpt.zero_grad()
# linearOpt.zero_grad()
# loss = F.mse_loss(y, torch.FloatTensor([0, 2.3, -1.33]))
# loss.backward()
# rnnOpt.step()
# linearOpt.step()

# print("AFTER")
# for param in rnn.parameters():
#     print(param[0])
#     break

# raise ValueError("ok")
# rnnOpt.zero_grad()
# linearOpt.zero_grad()
# loss = F.mse_loss(y, torch.FloatTensor([2.134, 4.9823, 5.33]))
# loss.backward()
# rnnOpt.step()
# linearOpt.step()



