import torch.nn as nn
from .TensorTools.TensorSVD import *
from .utilized import * 
from .lrr import *


 

class top_layer(nn.Module):
    def __init__(self, W, top_R, is_trainable,bias, TrainB=True):
        super(top_layer, self).__init__()
        U, S, VT = matrixSVD(W)
        A=U[:,:top_R] 
        B=torch.diag(S[:top_R])@VT[:top_R,:]
        self.r=top_R
        self.resW=W-A@B
        #self.resW=W 
        if is_trainable:
            for i in range(self.r):
                setattr(self,f'lorrasa_A_{i}',nn.Parameter(A[:,i]))
                setattr(self,f'lorrasa_B_{i}',nn.Parameter(B[i,:]))
        else:
            for i in range(self.r):
                setattr(self,f'lorrasa_A_{i}',A[:,i])  
                setattr(self,f'lorrasa_B_{i}',B[i,:])
         
 
        if TrainB==None:
            self.bias = torch.Tensor(bias).float()
            print("wwwwwwwwwwwwwwwww")
        else:  
            self.bias=nn.Parameter(torch.Tensor(bias).float())

    def forward(self,x):
        
        all_A = torch.stack([getattr(self, f'lorrasa_A_{i}') for i in range(self.r)], dim=1)  # 形状: (r-1, output_dim)
        all_B = torch.stack([getattr(self, f'lorrasa_B_{i}') for i in range(self.r)], dim=0)  # 形状: (r-1, input_dim)
        
       
        
        x1=nn.functional.linear(x,all_B) 
        resx=nn.functional.linear(x,self.resW.to(dtype=x.dtype)) 

        if self.bias is not None:
            x2=nn.functional.linear(x1,all_A,self.bias)
        else:
            x2=nn.functional.linear(x1,all_A) 
            
        return x2+resx

 
 

class top_layer_bias(nn.Module):
    def __init__(self,  W, top_R,is_trainable):
        super(top_layer_bias, self).__init__()
        U, S, VT = matrixSVD(W)
        A=U[:,:top_R] 
        B=torch.diag(S[:top_R])@VT[:top_R,:]
        self.r=top_R
        self.resW=W-A@B 
        if is_trainable: 
            for i in range(self.r):
                setattr(self,f'lorrasa_A_{i}',nn.Parameter(A[:,i]))
                setattr(self,f'lorrasa_B_{i}',nn.Parameter(B[i,:])) 
            
        else: 
            for i in range(self.r):
                setattr(self,f'lorrasa_A_{i}',A[:,i])  
                setattr(self,f'lorrasa_B_{i}',B[i,:]) 
         
  

    def forward(self,x): 
        
        all_A = torch.stack([getattr(self, f'lorrasa_A_{i}') for i in range(self.r)], dim=1)  # 形状: (r-1, output_dim)
        all_B = torch.stack([getattr(self, f'lorrasa_B_{i}') for i in range(self.r)], dim=0)  # 形状: (r-1, input_dim)
        
        oness = torch.ones(*x.shape[:-1], 1, device=x.device)
        newx = torch.cat((x, oness), dim=-1)  
        x1=nn.functional.linear(newx,all_B) 
        x2=nn.functional.linear(x1,all_A) 

        resx=nn.functional.linear(newx,self.resW.to(dtype=newx.dtype)) 
        
        return x2+resx


class top_layer_bias0(nn.Module):
    def __init__(self,W,top_R,is_trainable):
        super(top_layer_bias0, self).__init__()
        U, S, VT = matrixSVD(W)
        A=U[:,:top_R] 
        B=torch.diag(S[:top_R])@VT[:top_R,:]
 
        self.resW=W-A@B 
        if is_trainable:
            setattr(self,f'lorrasa_A',nn.Parameter(A))
            setattr(self,f'lorrasa_B',nn.Parameter(B))
        else:
            setattr(self,f'lorrasa_A',A)  
            setattr(self,f'lorrasa_B',B)
         
  

    def forward(self,x): 
        #print(x.shape)
       
        #ones = torch.ones(x.shape[0], x.shape[1], 1, device=x.device)
        oness = torch.ones(*x.shape[:-1], 1, device=x.device)
        newx = torch.cat((x, oness), dim=-1)
        A=getattr(self,f'lorrasa_A')
        B=getattr(self,f'lorrasa_B') 
        x1=nn.functional.linear(newx,B) 
        x2=nn.functional.linear(x1,A) 

        resx=nn.functional.linear(newx,self.resW.to(dtype=newx.dtype)) 
          
        return x2+resx
 
class replacing_Linear_with_topLinear(nn.Module):
    def __init__(self,args, model, module_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super(replacing_Linear_with_topLinear, self).__init__() 
        self.model=model
        self.top_R=args.top_R
        self.module_name=module_name
        self.with_bias=False
        self.device=device   
        if self.with_bias is True:
            self.replacing_Linear_weights_bias(self.model)
        else:
            self.replacing_Linear_weights(self.model)
         
        
         


    def replacing_Linear_weights(self, module):
        for name, layer in module.named_children():
            if isinstance(layer, nn.Linear):
                W =layer.weight.detach().cpu().numpy() 
                self.replace_layer(module,name,torch.tensor(W, device=self.device)) 
                #return model
            else:
                self.replacing_Linear_weights(layer) 
        return None
  
    def replacing_Linear_weights_bias(self, module): 
        for name, layer in module.named_children():
            if isinstance(layer, nn.Linear):
                weight=layer.weight.detach().cpu().numpy()
                #print(weight.shape)
                bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else np.zeros(weight.shape[0])
                #print(bias.shape)
                W = np.hstack((weight, bias[:, None])) 
                self.replace_layer(module,name,torch.tensor(W, device=self.device)) 
            else:
                self.replacing_Linear_weights_bias(layer)
        return None


 
     

    def get_nested_linear_iter(model):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                print(name)
                print(module.weight.detach().cpu().numpy().shape)
                #return model
            else:
                get_nested_linear_iter(module) 
        return None




    
    def get_nested_attr_iter(self,model, objname): 
        for name, module in model.named_children(): 
            if name==objname:
                return model
                
            else:
                result=self.get_nested_attr_iter(module,objname)
                if result is not None:
                    return result 
        return None

 
    def replace_layer(self, module,name,W):
        child=getattr(module, name)
        if name in self.module_name:
            is_trainable=True
        else:
            is_trainable=False

        if is_trainable:
            if self.with_bias is True:
                setattr(module, name, top_layer_bias(W,self.top_R,is_trainable)) 
            else:
                setattr(module, name, top_layer(W,self.top_R,is_trainable,torch.Tensor(child.bias).to(self.device) )) 

    def add_tensors(self,multi_heads_fa,name):
        W_tensor = self.lorrasa_Tensor 
        setattr(multi_heads_fa, name, self.lorrasa_Tensor) 
       
 
  
 
        

    
 