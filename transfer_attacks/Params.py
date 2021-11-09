"""
Parameters for IFSGM attack and CW attack that is used by the Transferer class.
"""

class IFSGM_Params():
    
    def __init__(self):
        
        # Attack Params
        self.batch_size = 500
        self.eps = 0.1
        self.alpha = 0.01
        self.iteration = 15
        self.target = 9
        self.x_val_min = 0
        self.x_val_max = 1
        
    def set_params(self, batch_size=None, eps=None, alpha=None, iteration = None,
                   target = None, x_val_min = None, x_val_max = None):
        
        if batch_size is not None:
            self.batch_size = batch_size
            
        if eps is not None:
            self.eps = eps
            
        if alpha is not None:
            self.alpha = alpha
            
        if iteration is not None:
            self.iteration = iteration
            
        if target is not None:
            self.target = target
            
        if x_val_min is not None:
            self.x_val_min = x_val_min
            
        if x_val_max is not None:
            self.x_val_max = x_val_max
            
class CW_Params():
    
    def __init__(self):
        
        # Attack Params
        self.batch_size = 500
        self.confidence = 20 # AKA Transferability metric
        self.optimizer_lr = 5e-4 
        self.iteration = 20
        self.target = 20
        self.x_val_mean = [0.5]
        self.x_val_std = [0.5]
        
        
    def set_params(self, batch_size=None, confidence=None, optimizer_lr=None, iteration = None,
                   target = None, x_val_mean = None, x_val_std = None):
        
        if batch_size is not None:
            self.batch_size = batch_size
            
        if confidence is not None:
            self.confidence = confidence
            
        if optimizer_lr is not None:
            self.optimizer_lr = optimizer_lr
            
        if iteration is not None:
            self.iteration = iteration
            
        if target is not None:
            self.target = target
            
        if x_val_mean is not None:
            self.x_val_mean = x_val_mean
            
        if x_val_std is not None:
            self.x_val_std = x_val_std
