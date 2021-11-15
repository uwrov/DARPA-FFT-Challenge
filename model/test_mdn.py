"""A script that shows how to use the MDN. It's a simple MDN with a single
nonlinearity that's trained to output 1D samples given a 2D input.
"""
import matplotlib.pyplot as plt
import sys
from MDN import MDN, sampling_test, mdn_loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

RNG = torch.Generator()
RNG.manual_seed(446)

input_dims = 2
output_dims = 2
num_gaussians = 4

total_num = 500

mu_matrix = torch.Tensor([[2,4], 
[0 , 0],
[1, 5],
[2, 2],
[-10, -3],
 ])

sigma1 = [4, 1, 2, 5, 10]
sigma2 = [2, 1, 4, 5, 10]
pho = [0.9, 0, 0.5, 0.2, 0.99]

sigma_matrix = []
for i in range(0, 5):
    sigma_matrix.append(
        torch.Tensor([[sigma1[i]**2, pho[i]*sigma1[i]*sigma2[i]], 
        [pho[i]*sigma1[i]*sigma2[i], sigma2[i]**2]])
        )

def translate_cluster(cluster, dim, amount):
    """Translates a cluster in a particular dimension by some amount
    """
    translation = torch.ones(cluster.size(0)) * amount
    cluster.transpose(0, 1)[dim].add_(translation)
    return cluster


print("Generating training data... ", end='')
'''
cluster1 = torch.randn((total_num, input_dims + output_dims)) / 4
cluster1 = translate_cluster(cluster1, 1, 1.2)
cluster2 = torch.randn((total_num, input_dims + output_dims)) / 4
cluster2 = translate_cluster(cluster2, 0, -1.2)
cluster3 = torch.randn((total_num, input_dims + output_dims)) / 4
cluster3 = translate_cluster(cluster3, 2, -1.2)
#cluster4 = torch.randn((total_num, input_dims + output_dims)) / 4
#cluster4 = translate_cluster(cluster4, 3, 1.5)
'''

CLUSTER_NUM = 4
w = [[1,1], [4,1], [2,2], [0.6, 0.3],[1, 1.2]]
bias = [[0,0], [2,10], [-5, -5], [7, -6], [7, -6]]
train_data = []

for i in range(0, CLUSTER_NUM):
    clusterx = w[i][0]*torch.randn((total_num, 1),generator = RNG) + bias[i][0]
    clustery =  w[i][0]*torch.randn((total_num, 1),generator = RNG) + bias[i][1]

    m = D.MultivariateNormal(mu_matrix[i], sigma_matrix[i])

    label = m.sample((total_num, ))
    train_data.append(torch.cat([clusterx, clustery, label], 1))
    

training_set = torch.cat(train_data, 0)
print(training_set.shape)
print('Done')
'''
fig = plt.figure(0)
ax = fig.add_subplot(projection='3d')

xs = training_set[:, 0]
ys = training_set[:, 1]
zs = training_set[:, 3]

ax.scatter(xs, ys, zs, label='target')
ax.legend()

plt.show()
'''
print("Initializing model... ", end='')
model = nn.Sequential(
    nn.Linear(input_dims, 5),
    nn.ReLU(),
    MDN(5, 1, num_gaussians)
)

optimizer = optim.Adam(model.parameters(), lr=0.005)
print('Done')

print('Training model... \n', end='')
sys.stdout.flush()
for epoch in range(3000):
    model.zero_grad()
    index = torch.randperm(training_set.size(0))
    pi, sigma, mu, pho = model(training_set[:, 0:input_dims])
  
    #print(pi, sigma, mu, pho)
    loss = mdn_loss(pi, mu,sigma, pho , training_set[:, input_dims:])
    loss.backward()
    optimizer.step()
    if epoch % 100 == 99:
        print(f' {round(epoch/10)}%', end='')
        print(loss.item())
        sys.stdout.flush()
print(' Done')


print('Generating samples... ', end='')
pi, sigma, mu, pho = model(training_set[:, 0:input_dims])

print(pi)

samples = sampling_test(pi, sigma, mu, pho)
print('Done')

print('Saving samples.png... ', end='')
fig = plt.figure(0)
ax = fig.add_subplot(projection='3d')

xs = training_set[:, 0]
ys = training_set[:, 1]
zs = training_set[:, 2]
zs1 = training_set[:, 3]

ax.scatter(xs, ys, zs, label='target', s=2)
ax.scatter(xs, ys, samples[:, 0], label='samples', s=2)
ax.legend()
fig.savefig('samples.png')
print('Done')

fig = plt.figure(1)
ax = fig.add_subplot(projection='3d')

xs = training_set[:, 0]
ys = training_set[:, 1]
zs = training_set[:, 2]
zs1 = training_set[:, 3]

ax.scatter(xs, ys, zs1, label='target', s=2)
ax.scatter(xs, ys, samples[:,1], label='samples', s=2)
ax.legend()
fig.savefig('samples1.png')
print('Done')
plt.show()