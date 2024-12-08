 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
from typing import Optional, List 
from .gm_bcd import *

class SubspacesAllocator(object):
    def __init__(self, model,target_KK, importnodes_num,init_warmup:int, final_warmup:int, mask_interval:int,
                 beta1:float, beta2:float, total_step:Optional[int]=None):
        super(SubspacesAllocator, self).__init__()
        #相比较秩的大小我们更关心哪些子空间需要被更新 
        self.target_KK = target_KK
        self.importnodes_num = importnodes_num
        self.importnodes_indices=None
        self.initial_warmup = init_warmup 
        self.final_warmup = final_warmup 
        self.mask_interval = mask_interval 
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step

        self.model = model
        self.ipt = {} 
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {} 
        self.score_point=[]
        #self.get_lorrasa_param_name()

       # self.tb_writter = tb_writter
        #self.log_interval = tb_writter_loginterval 

        assert (self.beta1<1 and self.beta1>0)
        assert (self.beta2<1 and self.beta2>0)
        
    def set_total_step(self, total_step:int): 
        # Set total step number 
        self.total_step = total_step
        #print(self.total_step)
        #print(self.initial_warmup+self.final_warmup)
        assert (self.total_step>self.final_warmup and self.final_warmup>self.initial_warmup)
         

  

    def schedule_threshold(self,total_KK, step:int):
        # Global budget schedule
        mask_ind = False 
        target_KK = self.target_KK 
        initial_warmup = self.initial_warmup 
        final_warmup = self.final_warmup 
        total_step = self.total_step 
        self.global_step = step
        if step < initial_warmup: 
            # Initial warmup 
            curr_KK = total_KK
            mask_ind = False 
            is_collect=False

        elif step >= initial_warmup and step < final_warmup:
            curr_KK = total_KK
            mask_ind = False 
            is_collect=True
            
        elif step == final_warmup: 
            # Final fine-tuning 
            curr_KK = self.target_KK  
            mask_ind = True
            is_collect=False
        else:  
            curr_KK = self.target_KK 
            mask_ind = False 
            is_collect=False
        return curr_KK, mask_ind,is_collect
        

    def update_ipt(self, model): 
        for n,p in model.named_parameters():
            if "lorrasa_" in n and p.grad is not None: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt[n] = (p * p.grad).abs().detach() #magnitude of the gradient-weight product
                    # Update sensitivity (9)
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty (10)
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()

            if "bias" in n and p.grad is not None: 
                if n not in self.ipt:
                    self.ipt[n] = torch.zeros_like(p)
                    self.exp_avg_ipt[n] = torch.zeros_like(p) 
                    self.exp_avg_unc[n] = torch.zeros_like(p) 
                with torch.no_grad():
                    # Calculate sensitivity 
                    self.ipt[n] = (p * p.grad).abs().detach() #magnitude of the gradient-weight product
                    # Update sensitivity (9)
                    self.exp_avg_ipt[n] = self.beta1 * self.exp_avg_ipt[n] + \
                                        (1-self.beta1)*self.ipt[n]
                    # Update uncertainty (10)
                    self.exp_avg_unc[n] = self.beta2 * self.exp_avg_unc[n] + \
                                        (1-self.beta2)*(self.ipt[n]-self.exp_avg_ipt[n]).abs()

    def calculate_score(self, n, p=None, metric="ipt"):
        if metric == "ipt":
            # Combine the senstivity and uncertainty (11)
            ipt_score = self.exp_avg_ipt[n] * self.exp_avg_unc[n]
        elif metric == "mag":
            ipt_score = p.abs().detach().clone() 
        else:
            raise ValueError("Unexcptected Metric: %s"%metric)
        return ipt_score 

    # def _combine_ipt(self,   ipt_AB):
    #     print(ipt_AB)
    #     ipt_AB = ipt_AB.sum(dim=1, keepdim=False)
    #     sum_ipt =  ipt_AB.view(-1)
    #     print(sum_ipt)
    #     print('123123123123123HHH')
    #     return sum_ipt 

    def collect_scores(self,  model):
        self.update_ipt(model) 
        is_dict = {}
        combine_dict = {}  
        # Calculate the importance score for each sub matrix 
        for n,p in model.named_parameters(): 
            if "lorrasa_A" in n and p.grad is not None: 
 
                hdim_a=0
                #print(p.requires_grad)
                #print(n)
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score)
                name_mat = n.replace("lorrasa_A", "%s")
                #print(name_mat) 
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                    #print(combine_dict[name_mat])
                else:
                    combine_dict[name_mat].append(comb_ipt)
               
            if "lorrasa_B" in n and p.grad is not None: 
                # hdim_b, rdim = p.shape  
                ipt_score = self.calculate_score(n, metric="ipt")
                comb_ipt = torch.mean(ipt_score) 
                name_mat = n.replace("lorrasa_B", "%s")
                #print(p.requires_grad)
                #print(n)
                #print(name_mat)
   
                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                    #print(combine_dict[name_mat])
                else:
                    combine_dict[name_mat].append(comb_ipt) 
                    #print(combine_dict[name_mat])

            if "bias" in n and p.grad is not None:
                ipt_score = self.calculate_score(n, metric="ipt") 
                comb_ipt = torch.mean(ipt_score)
                name_mat = n 

                if name_mat not in combine_dict: 
                    combine_dict[name_mat] = [comb_ipt]
                    #print(combine_dict[name_mat])
                else:
                    combine_dict[name_mat].append(comb_ipt) 
                

        # Combine the importance scores 
        #all_is = []
        k=0
        for name_mat in combine_dict:
            k=k+1
              
            #ipt_AB = torch.cat(combine_dict[name_mat], dim=1)
            if "bias" in name_mat:
                sum_ipt = combine_dict[name_mat][0]
            else:
                sum_ipt = (combine_dict[name_mat][0]+combine_dict[name_mat][1])/2#self._combine_ipt(ipt_AB) 
            is_dict[name_mat] = sum_ipt 
            #all_is.append(sum_ipt.view(-1)) 

        self.score_point.append(is_dict)

        return k

        
        # 这里假设 curr_KK > self.total_KK:
        #print(torch.cat(all_is))
        #print(curr_KK)
        #print(torch.cat(all_is).shape)
    def update_ggm(self,cur_KK):
        
        datas=[]
        name_list=[]
        ss=True
        for point in self.score_point:
            data0=[]
            for name_mat in point:
                sum_ipt=point[name_mat]
                data0.append(sum_ipt.view(-1))
               # print(sum_ipt.view(-1))
      
                if ss:
                    name_list.append(name_mat)
            ss=False
             
            data=torch.cat(data0)
            datas.append(data)
   

     
        SamCovM, SamMean= self.sample_covariance(torch.stack(datas, dim=0))
        self.importnodes_indices=torch.topk(SamMean,self.importnodes_num, largest=True).indices.tolist()[0]

        #return top_k_indices.tolist()
        
        bcd = BlockCoordinateDescent(
            SamCovM=SamCovM,
            importnodes_indices=self.importnodes_indices,
            lambda0=1000,
            tau=20,
            p=0.9,
            max_iter=50,
            max_iter_sub1=1,
            tol=1e-6,
            eta=0.0001,
            device='cpu'
            )
        Omega, _ = bcd.BCD_main()
        max_norm_columns = self.columns_with_max_norm(Omega,cur_KK)
        print(123123123123)
        print(max_norm_columns)
        chosen_points = [name_list[i] for i in max_norm_columns]
     
        return  chosen_points

    def update_model(self,  model,chosen_points): 
     
       
        # Mask out unimportant subspaces untrainable  
        with torch.no_grad(): 
            for n,p in model.named_parameters():
                 if "lorrasa" in n  and p.grad is not None:
                     if "lorrasa_A" in n:
                         name_mat = n.replace("lorrasa_A", "%s")
                     else:
                         name_mat = n.replace("lorrasa_B", "%s")
                         
                     if name_mat in chosen_points:
                         p.requires_grad = True 
                     else:
                         p.requires_grad = False  
                         p.grad =None

                 if "bias" in n and p.grad is not None:
                     if n in chosen_points:
                         p.requires_grad = True 
                     else:
                         p.requires_grad = False  
                         p.grad =None
                         

                     

        return   model

 

    def columns_with_max_norm(self, matrix,cur_KK):
        """
        Find the column indices with the maximum norm (excluding diagonal elements).

        Parameters:
        matrix (torch.Tensor): Symmetric matrix of shape (n, n).

        Returns:
        List[int]: Indices of columns with the maximum norm.
        """
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square."
        #assert torch.equal(matrix, matrix.T), "Matrix must be symmetric."

        n = matrix.shape[1]
        column_norms = []
        matrix0=matrix[self.importnodes_indices, :].clone()

        # Compute the norm for each column excluding diagonal elements
        for j in range(n):
            if j in self.importnodes_indices:
                column_norms.append(10000000000.0)
                continue 
            
            # Exclude diagonal by setting it to 0
            #column = matrix0[:, j].clone() 
            column_norm = torch.norm(matrix0[:, j], p=2)  # Compute L2 norm of the column
            column_norms.append(column_norm)

        # Convert to a tensor for easier processing
        column_norms = torch.tensor(column_norms)

      

        # Find all indices with the maximum norm
        top_k_indices = torch.topk(column_norms,cur_KK, largest=True).indices
        print(7777777777)
        print(column_norms[top_k_indices.tolist()])
        return top_k_indices.tolist()

 
 


    def sample_covariance(self,data):
        """
        Compute the sample covariance matrix for given data.

        Parameters:
        data (torch.Tensor): Input data of shape (n_samples, n_features),
                         where each row is a sample and each column is a feature.

        Returns:
        torch.Tensor: Sample covariance matrix of shape (n_features, n_features).
        """
        # Ensure the data is a PyTorch tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
        data=torch.log(data)#*(10**10) 
        data[data == float('-inf')] = -200.0
        data[data == float('inf')] = 200.0
        print(data)
        row_norms = torch.norm(data, p=2, dim=1, keepdim=True) 
        data = data / row_norms
        

  

        # Compute the sample covariance matrix
        n_samples = data.shape[0]
        # Center the data (subtract the mean of each column)
        mean_data=torch.mean(data, dim=0, keepdim=True)
     
        if n_samples==1:
            data_centered=data
        else:
            data_centered = data - mean_data
 
        if n_samples ==1:
            cov_matrix = torch.mm(data_centered.T, data_centered) / n_samples
        else: 
            cov_matrix = torch.mm(data_centered.T, data_centered) / (n_samples - 1)
        #print(cov_matrix)
        return cov_matrix,mean_data

    def update_and_mask(self, model, total_KK, global_step):
  
        # Budget schedule
        curr_KK, mask_ind,is_collect = self.schedule_threshold(total_KK, global_step)


        #curr_KK=self.target_KK
        #print(total_KK)
        #if global_step>2:
         #   self.update_ipt(model) 
         #   self.collect_scores(model)
         
        if mask_ind:   
            # Mask to target budget  
            if curr_KK <=total_KK: 
                chosen_points=self.update_ggm(curr_KK)
                model=self.update_model(model,chosen_points) 
                _=self.print_trainable_params(model)

                del self.ipt 
                del self.exp_avg_ipt
                del self.exp_avg_unc
                del self.cat_ipt
                del self.score_point 
                del self.importnodes_indices
        
                #self.ipt = {} 
                #self.exp_avg_ipt = {}
                #self.exp_avg_unc = {}
                #self.cat_ipt = {} 
                #self.score_point=[]
                
                
                

   
        return curr_KK, is_collect, model


    # def update_and_mask(self, model, total_KK, global_step):
    #     #if global_step<self.total_step-self.final_warmup and self.initial_warmup<=global_step:
    #         # Update importance scores element-wise 
    #     #self.update_ipt(model)
    #         # do not update ipt during final fine-tuning 
    #     # Budget schedule
    #     curr_KK, mask_ind,is_collect = self.schedule_threshold(total_KK, global_step)
    #     #mask_ind=True
    #     #is_collect=True
    #     #curr_KK=10
    #     #print(total_KK)
    #     if global_step<self.total_step-self.final_warmup:
    #         self.update_ipt(model)
    #         if is_collect:
    #             self.collect_scores(model)
    #             #print(self.score_point)
    #     #mask_ind=True
    #     if mask_ind:   
    #         # Mask to target budget  
    #         if curr_KK <=total_KK:
    #             chosen_points=self.update_ggm(curr_KK)
    #             model=self.update_model(model,chosen_points)
    #             #mask_threshold, model = self.mask_to_target(curr_KK,model) 
    #             total_KK=self.print_trainable_params(model)
    #         else:
    #             chosen_points = None
    #     else:
    #         chosen_points = None  
    #     return total_KK, chosen_points, model

    def print_trainable_params(self,model):
        trainable_params = 0
        all_param = 0
        KK=0
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                KK=1+KK 
            if 'classifier' in name:
                continue
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}")
        total_KK=KK//2
        return total_KK
 
 
 