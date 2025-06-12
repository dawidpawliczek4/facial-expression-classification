class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, monitor="val_loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_score = None
        self.counter = 0
        self.best_model_weights = None
        self.early_stop = False
    
    def __call__(self, model, metrics):
        score = metrics[self.monitor]
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return False
        
        if self.monitor == "val_loss":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                self.save_checkpoint(model)
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                self.save_checkpoint(model)
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop
    
    def save_checkpoint(self, model):
        self.best_model_weights = model.state_dict().copy()
    
    def load_best_model(self, model):
        model.load_state_dict(self.best_model_weights)