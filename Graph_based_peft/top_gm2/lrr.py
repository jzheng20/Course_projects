import torch
import scipy.io
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize



import numpy as np


def matrixSVD(tensor,dtype=torch.float32): 
    tensor=torch.tensor(tensor, dtype=dtype)
    U, S, V = torch.svd(tensor)  
    return U, S, V.T


def mPCA(tensor,r,dtype): 
    
    tensor=torch.tensor(tensor, dtype=dtype)
    U, S, V = torch.svd(tensor)  
    SS=torch.diag(S)
    return U[:,0:r], SS[0:r,0:r], V[:,0:r].T


def slicePCA(tensor,r,dtype=torch.float32):  
    tensor=torch.tensor(tensor, dtype=torch.float32)
    newTensor=torch.zeros(tensor.shape[0],tensor.shape[1],tensor.shape[2],tensor.shape[3]) 
    UU=torch.zeros(tensor.shape[0],tensor.shape[1],tensor.shape[2],r)
    SS=torch.zeros(tensor.shape[0],tensor.shape[1],r,r)
    VVT=torch.zeros(tensor.shape[0],tensor.shape[1],r,tensor.shape[3]) 
    
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            U, S, V = torch.svd(tensor[i,j,:,:])
            #print(len(S[0:r]))
            SS[i,j,:,:]=torch.diag(S[0:r])
            UU[i,j,:,:]=U[:,0:r]
            VVT[i,j,:,:]=V[:,0:r].T
            newTensor[i,j,:,:]=UU[i,j,:,:]@SS[i,j,:,:]@VVT[i,j,:,:]
    return UU.to(dtype), SS.to(dtype), VVT.to(dtype), newTensor.to(dtype)

def tensor_T(X):
    XT=torch.zeros(X.shape[0],X.shape[1],X.shape[3],X.shape[2]) 
    print(XT.shape)
    print(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            XT[i,j,:,:]=X[i,j,:,:].T
    return XT

def seg_tensor(X,rr,K,EK,dtype=torch.float32):
    
    XT= tensor_T(X)
    UU, SS, VVT, newXT=slicePCA(XT,rr,dtype)
    newX=tensor_T(newXT) 
    KK=[]
    ZZ=torch.zeros(XT.shape[0],XT.shape[3],XT.shape[3])
    indxx=[]
    for i in range(X.shape[0]):
        VT=VVT[i,0,:,:]
        ZZ[i,:,:] = VT.T@VT
        U, S, V = torch.svd(ZZ[i,:,:]) 
        S = torch.diag(S) 
        r = torch.sum(S > 1e-4 * S[0,0]).item()
        U = U[:, :r]
        S = S[:r, :r]
        U = torch.mm(U, torch.sqrt(S))
        U = torch.tensor(normalize(U.numpy(), norm='l2'))

        L = torch.mm(U, U.T).pow(4)

        # 谱聚类
        D = torch.diag(1.0 / torch.sqrt(torch.sum(L, dim=1)))
        L = torch.mm(torch.mm(D, L), D)
        U, S, V = torch.svd(L)
        if EK == "True":
            k0=torch.sum(S < 1e-2 * S[0]).item()
            print(23232323232)
            if k0>K:
                KK.append(k0)
            else:
                KK.append(K)
            #if k0 == 0:
             #   KK.append(K) 
              #  print("Single Subspace, should increseS the th")
            #else:
             #   KK.append(k0) 
        else:
            KK.append(K) 
        V = U[:,0:KK[i]]
        V = torch.mm(D, V)
        # K-means  
        kmeans = KMeans(n_clusters=KK[i], init='random', n_init=1, max_iter=300, random_state=123456789)
        idx = kmeans.fit_predict(V.numpy())
        indxx.append(idx)
    return indxx, ZZ.to(dtype),newX,KK  


 

def one_th(tensor,th):
    n1=tensor.shape[0]
    n2=tensor.shape[1]
    new_tensor=torch.zeros(n1,n2)
    for i in range(n1):
        for j in range(n2):
            if torch.abs(tensor[i,j])>th:
                new_tensor[i,j]=1.0
    return new_tensor


 

def seg_locations(WW,index):
    locations=[]
    for ii in range(WW.shape[0]):
        W=WW[ii,:,:]
        # each collumn in W is a sample
        K = index[ii].max().item()+1
        location=[]
        for i in range(K):
            location.append(np.where(index[ii] == i)[0]) 
        locations.append(location)
    return locations





def A_in_B(A,B,K):
    AA=torch.tensor(A)
    BB=[]
    re=torch.zeros(K)
    reB=torch.zeros(K)
    for i in range(K):
        BB.append(torch.tensor(B[i]))
        reB[i]=len(torch.tensor(B[i]))
        
    result = [] 
        
    for a in AA:
        found = False
        for i, group in enumerate(BB):
            if a in group:
                result.append(i)
                re[i]=1+re[i]
                found = True
                break 
                
        if not found:
            result.append(None)
    result = torch.tensor(result)
    return result,re,reB


 
     
  
 
def get_trainable_subspaces(trainable_rows,seg_results,KK,p):
    top_indicess=[]
    for ii in range(len(trainable_rows)):
        result,re,reB=A_in_B(trainable_rows[ii],seg_results[ii],KK[ii]) 
        top_indices=[]
        for i in range(KK[ii]):
            if re[i]/reB[i] > p:
                top_indices.append(i)
        top_indicess.append(top_indices)
    return top_indicess



def get_trainable_subspaces_all(num_layers,KK):
    top_indicess=[]
    for ii in range(num_layers):
        top_indices=[]
        for i in range(KK[ii]):
            top_indices.append(i)
        top_indicess.append(top_indices)
    return top_indicess