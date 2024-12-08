import torch.nn as nn
from .TensorTools.TensorSVD import *
from .utilized import * 
from .Replacing import * 

def get_nested_linear_iter(model,module_name):
    for name, module in model.named_children():
        if len(list(module.children()))==0:
            #print(1111111)
            if name not in module_name: 
                for  param in module.parameters():
                    #print(name)
                    #print(name0)
                    param.requires_grad=False
                
            #for name0, param in module.named_parameters():
             #   print(name)
               # print(name0)
              #  print(param.requires_grad)  
        else: 
            #if len(list(module.children()))==0:
                #print(name)
                #print(111111111)
                #for name0, param in module.named_parameters():
                   # print(name)
                    #print(name0)
                    #print(param.requires_grad)
                   # param.requires_grad=False
            #else:
                #print(2222222222)
            get_nested_linear_iter(module,module_name)
               #result=get_nested_linear_iter(module)
               # if result is not None:
               # return result 
    return None
 
 
 
def Loading(args,  model, r, module_name,device):
    #for param in model.parameters():
        #param.requires_grad = False 
     
    args.top_R=r#*len(layers_name[i]) 
    HH=replacing_Linear_with_topLinear(args, model,module_name,device)
    #get_nested_linear_iter(model,module_name)
        
    return model 
         

  
        
    
    
 