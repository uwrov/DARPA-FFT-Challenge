import numpy as np
import math 


from scipy.stats import multivariate_normal


mu1 = 0.1058
mu2 = -0.0240
p = 0#0.5065
inver_p = 1-p**2
sigma1 = 0.3454
sigma2 = 0.4193

x1 = 0.1375
x2 = -0.1050

Z = ((x1-mu1)/sigma1)**2 + ((x2-mu2)/sigma2)**2 - 2*p*(x1-mu1)*(x2-mu2)/(sigma1*sigma2)

Z1 = -Z/(2*inver_p)

out = math.exp(Z1)/(2*math.pi*sigma2*sigma1*math.sqrt(inver_p))
print(Z, Z1, out, math.log(out))


var = multivariate_normal(mean=[mu1,mu2], cov=[[sigma1**2,p*sigma1*sigma2],[p*sigma1*sigma2,sigma2**2]])
print(var.pdf([x1,x2]))