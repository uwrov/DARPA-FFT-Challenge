import torch
import torch.distributions as D
from matplotlib import pyplot as plt
import numpy as np

mu_matrix = torch.Tensor([[2,4], 
[0 , 0],
[-4, -2] ])

m1 = D.Categorical(torch.tensor([ 0.1, 0.3, 0.6 ]))
avg = 0.1*mu_matrix[0] + 0.3*mu_matrix[1] + 0.6 *mu_matrix[2]
print(avg, "avg")
sigma1 = [4, 1, 2]
sigma2 = [2, 1, 4]
pho = [0.9, 0, 0.5]

sigma_matrix = []
for i in range(0, 3):
    sigma_matrix.append(
        torch.Tensor([[sigma1[i]**2, pho[i]*sigma1[i]*sigma2[i]], 
        [pho[i]*sigma1[i]*sigma2[i], sigma2[i]**2]])
        )

indexes = m1.sample((50000, ))


sample = []
for i in range(0, indexes.shape[0]):
    idx = indexes[i].item()
    m = D.MultivariateNormal(mu_matrix[idx], sigma_matrix[idx])
    data = m.sample().numpy()
    sample.append(data)

sample = np.array(sample)
avg_sample = np.mean(sample, axis = 0)
print(data.shape)
print(avg_sample, "avg2")
plt.scatter(sample[:, 0], sample[:, 1], s=1)
plt.scatter(avg_sample[0], avg_sample[1], c='red', s=8)
plt.xlim([-20, 20])
plt.ylim([-20, 20])
plt.show()