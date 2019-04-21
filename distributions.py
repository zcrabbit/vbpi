import numpy as np 


class Normal(object):
    def __init__(self, dim):
        self.d = dim
    
    def sample(self, mu, sigma, n_particles=1):
        if n_particles > 1:
            return np.random.normal(loc=mu, scale=np.abs(sigma), size=(n_particles, self.d))
        else:
            return np.random.normal(loc=mu, scale=np.abs(sigma))
            
    def log_prob(self, x, mu, sigma, grad=False):
        ratio = np.exp(np.log((x-mu)**2) - np.log(sigma**2))
        if not grad:
            return np.sum(-0.5*np.log(2*np.pi) - 0.5*np.log(sigma**2) - .5*ratio, axis=1)
        else:
            return ratio / (mu-x)
            
    def reparam_grad(self, x, mu, sigma):
        std_x = (x-mu) / sigma
        return np.ones(mu.shape), std_x
    
    def lp_param_grad(self, x, mu, sigma):
        ratio = np.exp(np.log((x-mu)**2) - np.log(sigma**2))
        return ratio/(x-mu), -1./sigma * (1.0 - ratio)