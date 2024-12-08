


 
import torch 
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct 

 
def matrixSVD(tensor): 
    tensor=torch.tensor(tensor, dtype=torch.float32)
    U, S, V = torch.svd(tensor)  
    return U, S, V.T

def apply_dft_transform(tensor, axis,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return torch.fft.fft(tensor, dim=axis)

def apply_dct_transform(tensor, axis, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    return torch.tensor(dct(tensor.to(device).numpy(), axis=axis), dtype=torch.float32)

 
 
def GetTD_TSVD(TT):
    TTsize=TT.shape
    # # Mode 4 Unfolding 
    # unfolded_tensor_1 = TT.reshape(-1, TT.shape[-1]) 
    # U4, S1, V1T=matrixSVD(unfolded_tensor_1.T)
    # unfolded_tensor_1 = torch.diag(S1)@V1T 
    # TT1=unfolded_tensor_1.T.reshape(TTsize[0], TTsize[1], TTsize[2], TTsize[3]) 

    # # Mode 3 Unfolding 
    # unfolded_tensor_2 = TT1.permute(3, 0, 1, 2).reshape(-1, TT1.shape[2]) 
    # U3, S2, V2T=matrixSVD(unfolded_tensor_2.T)
    # unfolded_tensor_2 = torch.diag(S2)@V2T 
    # TT2=unfolded_tensor_2.T.reshape(TTsize[3], TTsize[0], TTsize[1],TTsize[2])
    # TT2=TT2.permute(1, 2, 3, 0)

    # Mode 2 Unfolding 
    unfolded_tensor_3 = TT.permute(2, 3, 0, 1).reshape(-1, TT.shape[1]) 
    U2, S3, V3T=matrixSVD(unfolded_tensor_3.T)
    unfolded_tensor_3 = torch.diag(S3)@V3T 
    TT3=unfolded_tensor_3.T.reshape(TTsize[2], TTsize[3],TTsize[0],TTsize[1])
    TT3=TT3.permute(2, 3, 0, 1)

    # Mode 1 Unfolding 
    unfolded_tensor_4= TT3.permute(1, 2, 3, 0).reshape(-1, TT3.shape[0]) 
    U1, S4, V4T=matrixSVD(unfolded_tensor_4.T)
    unfolded_tensor_4 = torch.diag(S4)@V4T 
    TT4=unfolded_tensor_4.T.reshape(TTsize[1],TTsize[2],TTsize[3],TTsize[0])
    TT4=TT4.permute(3, 0, 1, 2)
    
    GG=TT4

    return GG, U1, U2

def GetTD_TSVD_mixed(TT):
    TTsize=TT.shape
    # # Mode 4 Unfolding 
    # unfolded_tensor_1 = TT.reshape(-1, TT.shape[-1]) 
    # U4, S1, V1T=matrixSVD(unfolded_tensor_1.T)
    # unfolded_tensor_1 = torch.diag(S1)@V1T 
    # TT1=unfolded_tensor_1.T.reshape(TTsize[0], TTsize[1], TTsize[2], TTsize[3]) 

    # # Mode 3 Unfolding 
    # unfolded_tensor_2 = TT1.permute(3, 0, 1, 2).reshape(-1, TT1.shape[2]) 
    # U3, S2, V2T=matrixSVD(unfolded_tensor_2.T)
    # unfolded_tensor_2 = torch.diag(S2)@V2T 
    # TT2=unfolded_tensor_2.T.reshape(TTsize[3], TTsize[0], TTsize[1],TTsize[2])
    # TT2=TT2.permute(1, 2, 3, 0)
   

    # Mode 1 Unfolding 
    unfolded_tensor_4= TT.permute(1, 2, 3, 0).reshape(-1, TT.shape[0]) 
    U1, S4, V4T=matrixSVD(unfolded_tensor_4.T)
    unfolded_tensor_4 = torch.diag(S4)@V4T 
    TT4=unfolded_tensor_4.T.reshape(TTsize[1],TTsize[2],TTsize[3],TTsize[0])
    TT4=TT4.permute(3, 0, 1, 2)
    
    GG=TT4

    U2 = torch.eye(TTsize[1],device=device)

    return GG, U1, U2

def GetTD_TSVD_tree(TT,device):
    TTsize=TT.shape 

    # Mode 2 Unfolding 
    unfolded_tensor_3 = TT.permute(2, 3, 0, 1).reshape(-1, TT.shape[1]) 
    U2, S3, V3T=matrixSVD(unfolded_tensor_3.T)
    unfolded_tensor_3 = torch.diag(S3)@V3T 
    TT3=unfolded_tensor_3.T.reshape(TTsize[2], TTsize[3],TTsize[0],TTsize[1])
    TT3=TT3.permute(2, 3, 0, 1)

    # Mode 1 Unfolding
    U1= torch.zeros((TTsize[0],TTsize[0],TTsize[1]),device=device)
    TT4=torch.zeros((TTsize[0],TTsize[1],TTsize[2],TTsize[3]),device=device)
    for i in range(TT.shape[1]):
        TT3i=TT3[:,i,:,:]
        unfolded_tensor_4i= TT3i.permute(1, 2, 0).reshape(-1, TT3i.shape[0]) 
        U1[:,:,i], S4, V4T=matrixSVD(unfolded_tensor_4i.T)
        unfolded_tensor_4i = torch.diag(S4)@V4T 
        TT4i=unfolded_tensor_4i.T.reshape(TTsize[2],TTsize[3],TTsize[0])
        TT4[:,i,:,:]=TT4i.permute(2, 0, 1)
    
    GG=TT4

    return GG, U1, U2


def GetTD_TSVD_tree_mix(TT,device):
    TTsize=TT.shape 

    # # Mode 4 Unfolding 
    # unfolded_tensor_1 = TT.reshape(-1, TT.shape[-1]) 
    # U4, S1, V1T=matrixSVD(unfolded_tensor_1.T)
    # unfolded_tensor_1 = torch.diag(S1)@V1T 
    # TT1=unfolded_tensor_1.T.reshape(TTsize[0], TTsize[1], TTsize[2], TTsize[3]) 

    # # Mode 3 Unfolding 
    # unfolded_tensor_2 = TT1.permute(3, 0, 1, 2).reshape(-1, TT1.shape[2]) 
    # U3, S2, V2T=matrixSVD(unfolded_tensor_2.T)
    # unfolded_tensor_2 = torch.diag(S2)@V2T 
    # TT2=unfolded_tensor_2.T.reshape(TTsize[3], TTsize[0], TTsize[1],TTsize[2])
    # TT2=TT2.permute(1, 2, 3, 0)

    # # Mode 2 Unfolding 
    # unfolded_tensor_3 = TT.permute(2, 3, 0, 1).reshape(-1, TT.shape[1]) 
    # U2, S3, V3T=matrixSVD(unfolded_tensor_3.T)
    # unfolded_tensor_3 = torch.diag(S3)@V3T 
    # TT3=unfolded_tensor_3.T.reshape(TTsize[2], TTsize[3],TTsize[0],TTsize[1])
    # TT3=TT3.permute(2, 3, 0, 1)
    U2 = torch.eye(TTsize[1],device=device)
    TT3=TT

    # Mode 1 Unfolding
    U1= torch.zeros((TTsize[0],TTsize[0],TTsize[1]),device=device)
    TT4=torch.zeros((TTsize[0],TTsize[1],TTsize[2],TTsize[3]),device=device)
    for i in range(TT.shape[1]):
        print(i)
        TT3i=TT3[:,i,:,:]
        unfolded_tensor_4i= TT3i.permute(1, 2, 0).reshape(-1, TT3i.shape[0]) 
        U1[:,:,i], S4, V4T=matrixSVD(unfolded_tensor_4i.T)
        unfolded_tensor_4i = torch.diag(S4)@V4T 
        TT4i=unfolded_tensor_4i.T.reshape(TTsize[2],TTsize[3],TTsize[0])
        TT4[:,i,:,:]=TT4i.permute(2, 0, 1)
    #U1[:,:,TTsize[1]-1]=torch.eye(TTsize[0],device=device)
    GG=TT4

    return GG, U1, U2

def tmprod(A,Q,k):
    #newTensor=torch.tensor(A.clone()) 
    Tensor=A.clone()
    reshaped_tensor,Tsize = mFolding(Tensor,k)
    newTensor=mUnfolding(torch.matmul(Q,reshaped_tensor),k,Tsize)
    return newTensor

def mFolding(A,k):
    Asize=A.size()
    h=A.ndimension()
    dims=tuple([x + k for x in range(h-k)])+tuple(range(k))
    Tensor=A.permute(dims)
    reshaped_tensor = Tensor.reshape(Asize[k], -1)
    return torch.tensor(reshaped_tensor,dtype=torch.float32), Tensor.size()

def mUnfolding(matrixA,k,nAsize): 
    h=len(nAsize)
    invdims=tuple([x + h-k for x in range(k)])+tuple(range(h-k))
    TensorA=matrixA.reshape(nAsize)
    return TensorA.permute(invdims)

def TensorDecomposition_TSVD(TT,TransformType,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    TTsize=TT.shape
    if TransformType=='DFT':
        TT1=apply_dft_transform(TT, 0)
        GG=apply_dft_transform(TT1, 1)  
        U1=apply_dft_transform(torch.eye(TTsize[0]), 1)  
        U2=apply_dft_transform(torch.eye(TTsize[1]), 1)         
    elif TransformType=='DCT':
        TT1=apply_dct_transform(TT, 0)
        GG=apply_dct_transform(TT1, 1)  
        U1=apply_dct_transform(torch.eye(TTsize[0]), 1)  
        U2=apply_dct_transform(torch.eye(TTsize[1]), 1)         
    elif TransformType=='HOSVD':
        GG, U1, U2=GetTD_TSVD(TT)
    elif TransformType=='mixed':
        GG, U1, U2=GetTD_TSVD_mixed(TT)
    elif TransformType=='TSVDtree':
        GG, U1, U2=GetTD_TSVD_tree(TT,device)
    elif TransformType=='TSVDtree_mix':
        GG, U1, U2=GetTD_TSVD_tree_mix(TT,device)
    else:
        GG=TT
        U1, U2 = torch.eye(TTsize[0],device=device),torch.eye(TTsize[1],device=device) 
    return GG, U1, U2  

 

def TensorSVD_4D(TT,TransformType='None'):
    TTsize=TT.shape
    GG, U1, U2=TensorDecomposition_TSVD(TT,TransformType)
    TTsize12_min=torch.min(torch.tensor(TTsize[2]),torch.tensor(TTsize[3]))
    #SS= np.zeros((TTsize12_min, TTsize[0]*TTsize[1]))
    SS= torch.zeros((TTsize[0],TTsize[1],TTsize12_min))
    UU= torch.zeros((TTsize[0],TTsize[1],TTsize[2],TTsize12_min))
    VVT= torch.zeros((TTsize[0],TTsize[1],TTsize12_min,TTsize[3]))
    t=0
    for ii in range(TTsize[0]):
        for jj in range(TTsize[1]):
            UU[ii,jj,:,:], SS[ii,jj,:], VVT[ii,jj,:,:]=matrixSVD(GG[ii,jj,:,:])
            t=t+1
    return UU, SS, VVT, GG, U1, U2 

 

 


def get_topkAB_4D(oTT,TT,component_number=1,TransformType='None',device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    TT=torch.tensor(TT,dtype=torch.float32)
    oTT=torch.tensor(oTT,dtype=torch.float32)
    TTsize=TT.shape
    Mzeros=torch.zeros((TTsize[2],TTsize[3]), device=device) 
    UU, SS, VVT, GG, U1, U2 = TensorSVD_4D(TT,TransformType=TransformType)
  
    AAs = []
    BBs = []
    GGs=torch.zeros(TTsize, device=device) 
    for ii in range(TTsize[0]):
        AAtemp=[]
        BBtemp=[] 
        for jj in range(TTsize[1]):
            num_ii_jj = component_number#(SS[ii,jj,:]>=topk_values_min).sum().item()#((topk_indices_3d[:, 0] == 
            AA=UU[ii,jj,:,:num_ii_jj]
            AAtemp.append(AA) 
            BB=torch.diag(SS[ii,jj,:num_ii_jj])@VVT[ii,jj,:num_ii_jj,:]
            BBtemp.append(BB)
            GGs[ii,jj,:,:]=AA@BB
        AAs.append(AAtemp) 
        BBs.append(BBtemp)
        
  
    W_main=torch.tensor(GGs)
    resW=oTT-W_main   
    return AAs, BBs, resW
