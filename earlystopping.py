#%%
import copy

#%%
class EarlyStopping():
    def __init__(self, patience = 5, threshold = 0, load_best_model = True, verbose = True):
        self.patience = patience
        self.threshold = threshold
        self.load_best_model = load_best_model
        self.verbose = verbose
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
        
    def __call__(self, model, current_loss):
        if not self.best_loss:
            self.best_loss = current_loss
            self.best_model = copy.deepcopy(model.state_dict())
        
        if self.best_loss - current_loss >= self.threshold:
            self.counter = 0
            self.best_loss = current_loss
            self.best_model.load_state_dict(model.state_dict()) 
        
        if self.best_loss - current_loss < self.threshold:
            self.counter += 1
            
            if self.counter >= self.patience:
                print(f'\nModel stopped at counter = {self.counter}')
                
                if self.load_best_model:
                    model.load_state_dict(self.best_model.state_dict())    
                return False
        
        print(f'\nCurrent counter: {self.counter}/{self.patience}')  
        return True
        
    