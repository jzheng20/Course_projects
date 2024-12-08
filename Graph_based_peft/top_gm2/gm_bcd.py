import torch
from typing import Tuple

class BlockCoordinateDescent:
    def __init__(
        self,
        SamCovM,
        importnodes_indices,
        lambda0: float = 1.0,
        tau: float = 1.0,
        p: int = 1,
        max_iter: int = 100,
        max_iter_sub1: int = 100,
        tol: float = 1e-6,
        eta: float = 0.01,
        device: str = 'cuda'
    ):
        """
        Initialize Block Coordinate Descent algorithm.
        
        Args:
            rho: Regularization parameter (> 0)
            tau: Additional parameter (> 0)
            p: Choice of norm (2 or np.inf)
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            device: 'cpu' or 'cuda' for PyTorch implementation
        """
        assert lambda0 > 0 and tau > 0, "rho and tau must be positive"
        assert p > 0, "p must be positive"



        self.importnodes_indices=importnodes_indices 
        self.device = SamCovM.device
        self.SamCovM=torch.tensor(SamCovM)
        #print(SamCovM.device)
        #print(torch.eye(SamCovM.shape[0],device=self.device).device)
        self.SamCovM_pinv = torch.linalg.pinv(SamCovM+0.0000000001*torch.eye(SamCovM.shape[0],device=self.device))

        # parameters for model 
        self.lambda0 = lambda0
        self.tau = tau
        self.p = p

        # parameters for main problem
        self.max_iter = max_iter
        self.tol = tol
    
        # parameters for subproblem 1
        self.max_iter_sub1 = 100
        self.eta = eta # Learning rate
        self.epsilon = 1e-6 

        # parameters for subproblem 2

   
        #Omega_next, Delta_next = self.BCD_main()

 

    def generalized_soft_thresholding(self, S, weight, p, J=5):
        """
        Generalized Soft Thresholding in PyTorch.
    
        Parameters:
        S (torch.Tensor): Input matrix or vector.
        weight (float or torch.Tensor): Regularization coefficient.
        p (float): Parameter of the norm.
        J (int): Number of iterations for iterative thresholding.

        Returns:
        torch.Tensor: The thresholded result.
        """


        S=torch.tensor(S, dtype=torch.float64)
        weight=torch.tensor(weight, dtype=torch.float64)
        # Ensure weight is a tensor with the same shape as S
        if weight.shape != S.shape:
            weight = torch.full_like(S, weight)
        
        weight = weight.flatten()

        # Flatten S for element-wise processing
        diagS = S.flatten()
        Delta = diagS.clone()
    
        # Compute sigma0 and tau_GST
        sigma0 = torch.abs(diagS)
        tau_GST = (2 * weight * (1 - p)) ** (1 / (2 - p)) + p * weight * (2 * (1 - p) * weight) ** ((p - 1) / (2 - p))
    
        # Iterate through each element
        for i in range(len(diagS)):
            if sigma0[i] > tau_GST[i]:
                delta = sigma0[i]
                for _ in range(J):  # Iterative thresholding
                    delta = sigma0[i] - weight[i] * p * (delta ** (p - 1))
                Delta[i] = torch.sign(diagS[i]) * delta
            else:
                Delta[i] = 0
    
        # Reshape Delta to the original shape of S
        Delta = Delta.reshape(S.shape)
    
        return Delta

 

    def generalized_soft_thresholding_col(self,X, tau, p):
        """
        Generalized Soft Thresholding for matrix columns in PyTorch.
    
        Parameters:
        X (torch.Tensor): Input matrix (each column will be thresholded).
        tau (float): Threshold parameter.
        p (float): Parameter of the norm.

        Returns:
        torch.Tensor: Thresholded matrix.
        """
        # Initialize output tensor



        Y =  X.clone()
        X0=X[self.importnodes_indices, :].clone()
        Y0=X0.clone() 
        #print(self.importnodes_indices)
        #print(X0.shape)
        #print(X.shape)
        #print(X0.shape)
        #print(X.shape[1])
    
        # Iterate over columns
        for i in range(X0.size(1)):  # Loop through each column
            if i in self.importnodes_indices:
                continue 
            column = X0[:, i].clone()
          
        
            # Compute Frobenius norm of the column
            column_norm = torch.norm(column, p=2)  # Frobenius norm for a vector is the L2 norm
        
            # Apply Generalized Soft Thresholding to the norm
            thresholded_norm = self.generalized_soft_thresholding(column_norm, tau, p)
        
            # Scale the column by the thresholded norm
            if column_norm > 0:  # Avoid division by zero
                Y0[:, i] = thresholded_norm / (column_norm + 1e-8) * column

        Y[self.importnodes_indices,:]=Y0.clone()
        #print(Y)
        return Y
   
        # Y = torch.zeros_like(X)
    
        # # Iterate over columns
        # for i in range(X.size(1)):  # Loop through each column
        #     column = X[:, i].clone()
        #     column[i] =0
        
        #     # Compute Frobenius norm of the column
        #     column_norm = torch.norm(column, p=2)  # Frobenius norm for a vector is the L2 norm
        
        #     # Apply Generalized Soft Thresholding to the norm
        #     thresholded_norm = self.generalized_soft_thresholding(column_norm, tau, p)
        
        #     # Scale the column by the thresholded norm
        #     if column_norm > 0:  # Avoid division by zero
        #         Y[:, i] = thresholded_norm / (column_norm + 1e-8) * column
        #     Y[i, i]=X[i, i].clone()
        # #print(Y)
        # return Y

 

    # # Objective function
    # def objective(Omega, A_s, lambda_reg):
    #     log_det_term = torch.logdet(Omega)
    #     trace_term = torch.trace(torch.mm(A_s, Omega))
    #     frobenius_norm = torch.norm(Omega, p='fro') ** 2
    #     return log_det_term - trace_term - lambda_reg * frobenius_norm
            
        # Gradient descent loop
    def Gradient_descent_sub1(self,A):
        A_s = 0.5 * (A + A.T) 
        Omega=A_s
    
        for iteration in range(self.max_iter_sub1):
            # Compute the inverse of Omega
            Omega_inv = torch.inverse(Omega+ 0.000000001*torch.eye(Omega.shape[0],device=self.device))
            
            # Gradient update rule
            gradient = (Omega_inv - A_s - 2 * self.lambda0 * Omega)/(2*self.lambda0)
            Omega_next = Omega + self.eta * gradient

            # Convergence check
            if torch.norm(Omega_next - Omega) < self.epsilon:
                print(f"Converged after {iteration + 1} iterations.")
                return Omega_next
                     

            # Update Omega
            Omega = Omega_next.clone()

        return Omega_next

    def BCD_main(self): 
        
        Omega=self.SamCovM_pinv 
   
        for iter in range(self.max_iter):
            print(iter)
           
            # update Delta
            Delta_next= self.generalized_soft_thresholding_col(Omega, self.tau/(2*self.lambda0), self.p)

            #print(Delta_next)
            
            # update Omega
            A=(self.SamCovM-2*self.lambda0*Delta_next)/(2*self.lambda0)
  
            Omega_next=self.Gradient_descent_sub1(A)

            
 
            
            if iter>1:
                if torch.max(torch.norm(Omega_next - Omega), torch.norm(Delta_next - Delta)) < self.tol:
                    print(f"Converged after {iter + 1} iterations.")
                    return Omega_next, Delta_next

            Omega = Omega_next.clone()
            Delta = Delta_next.clone()

        return Omega_next, Delta_next
        

 

 
