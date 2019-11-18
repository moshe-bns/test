import numpy as np
class BagsConstrainLoss:
    def __init__(self, constrains):
        self.constrains = constrains

    def myCustomLoss1(self, predictions,target, idx,device):
        np.random.shuffle(self.constrains)
        curr_loss = self.constrains[0].calc_loss(predictions,target, idx,device)
        for constrain in self.constrains[1:]:
            curr_loss = curr_loss + constrain.calc_loss(predictions,target, idx, device)
        self.loss = curr_loss
        return curr_loss

    def majority_loss(self, predictions,target, idx):
        num_constrains = len(self.constrains)
        loss = self.constrains[0].calc_loss(predictions,target, idx)
        for constrain in self.constrains[1:]:
            loss = loss + constrain.calc_loss(predictions,target, idx)
        self.loss = loss
        return loss

    def get_tensor_loss(self):
        return self.loss
