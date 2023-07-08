import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
import os, random
import yaml


# ===========================================================================================
# ======================================= NN Model ==========================================
# ===========================================================================================
class ConvAEIn(nn.Module):
    def __init__(self, params):
        super(ConvAEIn, self).__init__()
        kernelSize = params["kernelSize"]
        numHidden = params["numHidden"]

        self.batchSize = params["numSubj"] * params["numPerSubj"]

        self.padEncL1 = nn.ZeroPad2d((0, 2, 0, 2))
        self.encL1 = nn.Conv2d(1, numHidden[0], kernel_size=kernelSize[0], stride=2)

        self.padDecL1 = nn.ZeroPad2d((0, -1, 0, -1))
        self.decL1 = nn.ConvTranspose2d(numHidden[0], 1, kernel_size=kernelSize[0], stride=2)

    def forward(self, X):
        Z1 = F.relu(self.encL1(self.padEncL1(X)))

        output = F.relu(self.padDecL1(self.decL1(Z1)))
        return output
# ===========================================================================================
# ======================================= NN Model ==========================================
# ===========================================================================================

from torchvision import models
print("testing Resnet...")
class ResNet18(nn.Module):
    def __init__(self, in_channels=1, hidden=50):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(in_channels, 10, kernel_size=3, stride=2, padding=1, bias=False)
        self.relu = nn.LeakyReLU()

        # Replace residual blocks in layers 2, 3, and 4 with identity blocks
        self.resnet.layer1 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(),
            nn.Conv2d(20, 50, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(50)
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.relu(x)
        x = self.resnet.layer1(x) # Add the residual connection
        return x



class ConvAE(nn.Module):
    def __init__(self, params):
        super(ConvAE, self).__init__()
        kernelSize = params["kernelSize"]
        numHidden = params["numHidden"]
        cte = params["cte"]
        numSubj = params["numSubj"]
        rankEs = params["rankE"]

        self.batchSize = numSubj * params["numPerSubj"]

        # self.padEncL1 = nn.ZeroPad2d((0, 2, 0, 2))
        # self.encL1 = nn.Conv2d(1, numHidden[0], kernel_size=kernelSize[0], stride=2)
        self.resnet18 = ResNet18(in_channels=1)

        cc = np.zeros((self.batchSize, rankEs))

        self.C1 = nn.Parameter(Variable(torch.Tensor(cc), requires_grad=True))

        self.padDecL1 = nn.ZeroPad2d((0, -1, 0, -1))
        self.decL1 = nn.ConvTranspose2d(numHidden[0], 1, kernel_size=kernelSize[0], stride=2)

    def forward(self, X):
        Z1 = F.relu(self.resnet18(X))

        Y = (torch.matmul(self.C1, torch.transpose(self.C1, 0, 1))-torch.diag(torch.diag(torch.matmul(self.C1, torch.transpose(self.C1, 0, 1))))).mm(Z1.view(self.batchSize, -1))
        Y = Y.view(Z1.size())

        output = F.relu(self.padDecL1(self.decL1(Y)))
        return Z1, Y, self.C1, output
    


# ===========================================================================================
# ========================== Subspace Clustering Method DLRSC ==============================
# ===========================================================================================
def subspaceClusteringMLRDSC(images, params):
    numSubjects = params["numSubj"]
    numPerSubj  = params["numPerSubj"]
    numEpochs   = params["numEpochs"]
    fileName    = params["preTrainedModel"]
    alpha = params["alpha"]
    lr    = params["lr"]
    T     = params["T"]
    cte   = params["cte"]
    seedValue = params["seedValue"]
    batchSize = numSubjects * numPerSubj
    regparams = params["regparams"]
    label = params["label"]
    rankEs = params["rankE"]
    gamma_reg, lambda_reg = params["lambda"], params["gamma"]

    coilSubjects = images
    coilSubjects = coilSubjects.astype(float)
    labelSubjects = label
    labelSubjects = labelSubjects - labelSubjects.min() + 1
    labelSubjects = np.squeeze(labelSubjects)

    X = Variable(torch.Tensor(coilSubjects / 255.).cuda(), requires_grad=False)

    preTrainedMod = torch.load(fileName)
    CAE = ConvAE(params)
    CAE = CAE.cuda()

    parametersAE = dict([(name, param) for name, param in preTrainedMod.named_parameters()])

    for name, param in CAE.named_parameters():
        if name in parametersAE:
            param_pre  = parametersAE[name]
            param.data = param_pre.data

    if params["seedFlag"]:
        random.seed(seedValue)
        os.environ['PYTHONHASHSEED'] = str(seedValue)
        np.random.seed(seedValue)
        torch.manual_seed(seedValue)
        torch.cuda.manual_seed(seedValue)
        torch.cuda.manual_seed_all(seedValue)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    CAE.C1.data = (torch.Tensor(cte * np.random.randn(batchSize, rankEs))).cuda()

    optimizer = torch.optim.Adam(CAE.parameters(), lr=lr)
    numSamples = coilSubjects.shape[0]

    for epoch in range(numEpochs + 1):
        Z1, Y, C1, output = CAE(X)

        regLoss   = gamma_reg * (torch.norm(C1, p=2)**2)
        reconLoss =  (torch.norm(output - X, p=2) ** 2)
        expLoss   = lambda_reg * (torch.norm(Z1 - Y, p=2) ** 2)
        loss = reconLoss + regLoss + regparams * expLoss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch > 0 and epoch % T == 0:
            print("Losses  " + "Reconstruction: %.8f     Expression: %.8f     Regularization: %.8f" % (reconLoss / numSamples, expLoss, regLoss))
            mm1 = C1.detach().cpu().numpy()
            Coef = thrC(np.dot(mm1, mm1.T), alpha)
            yHat, _ = post_proC(Coef, labelSubjects.max(), params)
            errorClus = err_rate(labelSubjects, yHat)
            accuClus = 1 - errorClus
            print("Accuracy after %d" % (epoch), "Iterations: %.4f" % accuClus)

    return (1 - accuClus)

def best_map(L1, L2):
    # L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp

def post_proC(C, K, params):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    d = params["post_proc"][0]
    alpha = params["post_proc"][1]

    C = 0.5 * (C + C.T)
    C = C - np.diag(np.diag(C)) + np.eye(C.shape[0],C.shape[0])
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate

# ===========================================================================================
# ====================================== Main Function ======================================
# ===========================================================================================
if __name__ == "__main__":
    
    if torch.cuda.is_available():   
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: 
        device = torch.device('cpu')
    from torchinfo import summary
        
#     class ConvAEIn(nn.Module):
#         def __init__(self, params):
#             super().__init__()
#             kernelSize = params["kernelSize"]
#             numHidden = params["numHidden"]

#             self.batchSize = params["numSubj"] * params["numPerSubj"]

#             self.padEncL1 = nn.ZeroPad2d((0, 2, 0, 2))
#             self.encL1 = nn.Conv2d(1, numHidden[0], kernel_size=kernelSize[0], stride=2)

#             self.padDecL1 = nn.ZeroPad2d((0, -1, 0, -1))
#             self.decL1 = nn.ConvTranspose2d(numHidden[0], 1, kernel_size=kernelSize[0], stride=2)

#         def forward(self, X):
#             Z1 = F.relu(self.encL1(self.padEncL1(X)))

#             output = F.relu(self.padDecL1(self.decL1(Z1)))
#             return output
# # =======================
#     ae_in_params = {"kernelSize": [3],
#         "numHidden":  [50],
#         "input_size": [32, 32],
#         "numSubj": 100,
#         "numPerSubj": 72
#         } # pre-trained model

#     ae_in = ConvAEIn(params=ae_in_params)
#     print("done!")
    
    
#     from torchinfo import summary
#     summary(ae_in, input_size=(1, 32, 32), device=device,
#             col_names=["input_size","kernel_size", "output_size", "num_params"], col_width=20,
#             row_settings=["var_names"],depth=5)
    
#######################################
    print("Testing ConvAE...")
    
    ae_params = {"kernelSize": [3],
        "numHidden":  [50],
        "input_size": [32, 32],
        "numSubj": 8,
        "cte": 1.0e-4,
        "rankE": 10,
        "numPerSubj": 8
        } # pre-trained model
    
    class ConvAE(nn.Module):
        def __init__(self, params):
            super(ConvAE, self).__init__()
            kernelSize = params["kernelSize"]
            numHidden = params["numHidden"]
            cte = params["cte"]
            numSubj = params["numSubj"]
            rankEs = params["rankE"]

            self.batchSize = numSubj * params["numPerSubj"]

            print("batchSize:", self.batchSize)
            self.padEncL1 = nn.ZeroPad2d((0, 2, 0, 2))
            self.encL1 = nn.Conv2d(1, numHidden[0], kernel_size=kernelSize[0], stride=2)
            self.resnet18 = ResNet18(in_channels=1)

            cc = np.zeros((self.batchSize, rankEs))

            self.C1 = nn.Parameter(Variable(torch.Tensor(cc), requires_grad=True))
            print("C1Size:", self.C1.shape)
            self.padDecL1 = nn.ZeroPad2d((0, -1, 0, -1))
            self.decL1 = nn.ConvTranspose2d(numHidden[0], 1, kernel_size=kernelSize[0], stride=2)

        def forward(self, X):
            Z1 = F.relu(self.resnet18(X))
            print("Z1 shape:", Z1.shape)
            Y = (torch.matmul(self.C1, torch.transpose(self.C1, 0, 1))-torch.diag(torch.diag(torch.matmul(self.C1, torch.transpose(self.C1, 0, 1))))).mm(Z1.view(self.batchSize, -1))
            Y = Y.view(Z1.size())

            output = F.relu(self.padDecL1(self.decL1(Y)))
            return Z1, Y, self.C1, output
        
    ae = ConvAE(params=ae_params)
    dummy_input = torch.randn(64, 1, 32, 32)
    out = ae(dummy_input)
    print("out shape:", out[3].shape)
    
    
    summary(ae, input_size=(64,1, 32, 32), device=device,
            col_names=["input_size","kernel_size", "output_size", "num_params"], col_width=20,
            row_settings=["var_names"],depth=5)
    #########################
    
    # from torchvision import models
    # print("testing Resnet...")
    # class ResNet18(nn.Module):
    #     def __init__(self, in_channels=1):
    #         super().__init__()
    #         self.resnet = models.resnet18(weights=None)
    #         self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    #         self.relu = nn.LeakyReLU()
    #         # self.resnet.fc = nn.Linear(512, out_nodes)  # Flatten to 128 nodes
    #         # Replace residual blocks in layers 2, 3, and 4 with identity blocks
    #         self.resnet.layer2 = nn.Sequential(
    #             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    #             nn.BatchNorm2d(64),
    #             nn.LeakyReLU(),
    #             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
    #             nn.BatchNorm2d(64)
    #         )
    #         self.resnet.layer3 = nn.Sequential(
    #             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
    #             nn.BatchNorm2d(128),
    #             nn.LeakyReLU(),
    #             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
    #             nn.BatchNorm2d(128)
    #         )
    #         self.resnet.layer4 = nn.Sequential(
    #             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
    #             nn.BatchNorm2d(256),
    #             nn.LeakyReLU(),
    #             nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
    #             nn.BatchNorm2d(64))

    #     def forward(self, x):
    #         x = self.resnet.conv1(x)
    #         x = self.resnet.bn1(x)
    #         x = self.resnet.relu(x)
    #         #x = self.resnet.maxpool(x)

    #         x = self.resnet.layer1(x)
    #         x = self.resnet.layer2(x)
    #         x = self.resnet.layer3(x)
    #         x = self.resnet.layer4(x)

    #         return x
    
    
    # resnet = ResNet18(in_channels=1)
    # resnet.to(device=device)
    # resnet_dummmy = torch.randn(1, 1, 128, 128).float().to(device)
    # #resnet_dummmy.to(device)
    # res_out = resnet(resnet_dummmy)
    # print("res_out shape:", res_out.shape)
    
    # import torch
    # import torch.nn as nn
    # import torchvision.models as models
    # import torchinfo

    # # Load the ResNet-18 model
    # resnet18 = models.resnet18()

    # # Create a random input tensor for summary
    # dummy_input = torch.randn(1, 3, 224, 224)

    # # Print the model summary using torchinfo
    # summary = torchinfo.summary(resnet18, input_size=(1, 3, 224, 224),
    #                                         col_names=["input_size","kernel_size", "output_size", "num_params"], col_width=20,
    #          row_settings=["var_names"],depth=5)
    # print(summary)