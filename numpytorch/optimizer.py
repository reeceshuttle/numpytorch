class Optimizer:
    def __init__(self):
        raise NotImplementedError('Override me!')
    
    def Update(self, dLdZ):
        raise NotImplementedError('Override me!')

class VanillaOptimizer(Optimizer):
    def __init__(self, model):
        self.model = model

    def update(self, dLdZ):
        raise NotImplementedError('Implement me!')
        # will do the weight update here
    

class AdamOptimizer(Optimizer):
    def __init__(self, model):
        raise NotImplementedError('Implement me!')
        # during initialization will need to store first and second order momentum.