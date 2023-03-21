import torch
import torch.nn as nn
global mu
global var

class LinBnDrop(nn.Sequential):
    """Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"""
    def __init__(self, n_in, n_out, bn=True, p=0., act=None, lin_first=True):
        layers = [nn.BatchNorm1d(n_out if lin_first else n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

        
class Encoder(nn.Module):
    """Encoder for CITE-seq data"""
    def __init__(self, nfeatures_modality1=10703, hidden_modality1=185,  z_dim=128):
        super().__init__()
        self.nfeatures_modality1 = nfeatures_modality1
        self.encoder_modality1 = LinBnDrop(nfeatures_modality1, hidden_modality1, p=0.2, act=nn.ReLU())
        self.encoder = LinBnDrop(hidden_modality1, z_dim,  p=0.2, act=nn.ReLU())
        self.weights_modality1 = nn.Parameter(torch.rand((1,nfeatures_modality1)) * 0.001, requires_grad=True)
        self.fc_mu = LinBnDrop(z_dim,z_dim, p=0.2)
        self.fc_var = LinBnDrop(z_dim,z_dim, p=0.2)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, x):
        global mu
        global var
        x = self.encoder_modality1(x*self.weights_modality1)
        x = self.encoder(x)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        x = self.reparameterize(mu, var)
        return x
    

class Decoder(nn.Module):
    """Decoder for for 2 modalities data (citeseq data and shareseq data) """
    def __init__(self, nfeatures_modality1=10703, hidden_modality1=185, z_dim=128):
        super().__init__()
        self.nfeatures_modality1 = nfeatures_modality1
        self.decoder1 = LinBnDrop(z_dim, nfeatures_modality1, act=nn.ReLU())

    def forward(self, x):
        x = self.decoder1(x)
        return x

    
class CiteAutoencoder(nn.Module):
    def __init__(self, nfeatures_rna=0, hidden_rna=185, z_dim=20,classify_dim=17):
        """ Autoencoder for 2 modalities data (citeseq data and shareseq data) """
        super().__init__()
        self.encoder = Encoder(nfeatures_rna, hidden_rna, z_dim)
        self.classify = nn.Linear(z_dim, classify_dim)
        self.decoder = Decoder(nfeatures_rna, hidden_rna, z_dim)
        
    def forward(self, x):
        global mu
        global var
        x = self.encoder(x)
        x_cty = self.classify(x)
        x = self.decoder(x)
        return x, x_cty,mu,var
    
