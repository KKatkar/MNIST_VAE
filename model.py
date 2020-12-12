import torch
import torch.nn as nn
import torch.nn.functional as F

features = 16
# define a simple linear VAE
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
 
        # encoder
        # self.enc1 = nn.Linear(in_features=784, out_features=512)
        # self.enc2 = nn.Linear(in_features=512, out_features=features*2)
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc31 = nn.Linear(256, 2)
        self.fc32 = nn.Linear(256, 2)
 
        # decoder 
        # self.dec1 = nn.Linear(in_features=features, out_features=512)
        # self.dec2 = nn.Linear(in_features=512, out_features=784)

        self.fc4 = nn.Linear(2, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 784)
        
        #SAMPLING
    # def reparameterize(self, mu, log_var):
    #     """
    #     :param mu: mean from the encoder's latent space
    #     :param log_var: log variance from the encoder's latent space
    #     """
    #     std = torch.exp(0.5*log_var) # standard deviation
    #     eps = torch.randn_like(std) # `randn_like` as we need the same size
    #     sample = mu + (eps * std) # sampling as if coming from the input space
    #     return sample
 
    # def forward(self, x):
    #     # encoding
    #     x = F.relu(self.enc1(x))
    #     x = self.enc2(x).view(-1, 2, features)
    #     # get `mu` and `log_var`
    #     mu = x[:, 0, :] # the first feature values as mean
    #     log_var = x[:, 1, :] # the other feature values as variance
    #     # get the latent vector through reparameterization
    #     z = self.reparameterize(mu, log_var)
 
    #     # decoding
    #     x = F.relu(self.dec1(z))
    #     reconstruction = torch.sigmoid(self.dec2(x))
    #     return reconstruction, mu, log_var

    # def decoder(self,encoded):
    #     x=self.dec1(encoded)
    #     # get `mu` and `log_var`
    #     mu = x[:, 0, :] # the first feature values as mean
    #     log_var = x[:, 1, :] # the other feature values as variance
    #     # get the latent vector through reparameterization
    #     z = self.reparameterize(mu, log_var)
    #     x = F.relu(self.dec1(z))
    #     reconstruction = torch.sigmoid(self.dec2(x))
    #     return reconstruction, mu, log_var

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var