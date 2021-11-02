import torch
import torch.nn.functional as F
import torch.nn as nn



class Attack_methods(object):
    def __init__(self, attack_name, model, X, Y, epsilon, alpha, attack_iters):
        self.attack_name = attack_name
        self.model = model
        self.epsilon = epsilon
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.alpha = alpha
        self.opt = torch.optim.Adam(model.parameters(),lr = 0.01)
        self.attack_iters = attack_iters


    def choose_method(self):
        if self.attack_name == 'fgsm':
            return  self.fgsm()
        elif self.attack_name == 'pgd':
            return self.pgd()
        else:
            return self.nothing()

    def nothing():
        delta = torch.zeros_like(X)
        return delta

    def fgsm(self):
        delta = torch.zeros_like(self.X).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True
        output = self.model(self.X + delta)
        loss = nn.CrossEntropyLoss()(output, self.Y)
        loss.backward()
        grad =delta.grad.detach()
        delta.data = torch.clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)
        delta.data = torch.max(torch.min(1 - self.X, delta.data), 0 - self.X)
        delta = delta.detach()
        return delta
    
    def pgd(self):
        delta = torch.zeros_like(self.X).uniform_(-self.epsilon, self.epsilon)
        delta.data = torch.max(torch.min(1 - self.X, delta.data), 0 - self.X)
        
        for _ in range(self.attack_iters):
            delta.requires_grad = True
            output = self.model(self.X + delta)
            loss = nn.CrossEntropyLoss()(output, self.Y)
            self.opt.zero_grad()
            loss.backward()
            grad = delta.grad.detach()
            I = output.max(1)[1] == self.Y
            delta.data[I] = torch.clamp(delta + self.alpha * torch.sign(grad), -self.epsilon, self.epsilon)[I]
            delta.data[I] = torch.max(torch.min(1-self.X, delta.data), 0-self.X)[I]
        delta = delta.detach()
        return delta



if __name__ == '__main__':
    X = torch.tensor([1.0])
    Y = torch.tensor([1])
    output = torch.tensor([0.9])
    epsilon = 0.01
    alpha = 0.01
    model = torch.sin
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    #X.to(device)
    #Y.to(device)
    adversarial_attack = attack_methods('fgsm', model,  X, Y , epsilon, alpha)
    attack_method = adversarial_attack.choose_method()
