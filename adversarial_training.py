from  attack_method import Attack_methods
import torch
import torch.nn as nn
import time
import logging

class Adversarial_Trainings(object):
    def __init__(self, epochs, train_loader, model, opt, epsilon, alpha, iter_num, lr_max, lr_schedule, m_repeats, fname, logger):
        self.epochs = epochs
        self.train_loader = train_loader
        self.model = model
        self.opt = opt 
        self.epsilon = epsilon
        self.alpha = alpha
        self.iter_num = iter_num
        self.lr_max = lr_max 
        self.lr_schedule = lr_schedule
        self.m_repeats = m_repeats
        self.fname = fname
        self.logger = logger

    def fast_free_training(self): 
        for epoch in range(self.epochs):
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            m_repeats = 1

            for i, (X, y) in enumerate(self.train_loader):
                X, y = X.cuda(), y.cuda()
                lr = self.lr_schedule(epoch + (i+1)/len(self.train_loader))
                self.opt.param_groups[0].update(lr=lr)
                for _ in range(self.m_repeats): # if m_repeats = 1, fast; if m_repeats > 1 free.
                    # Generating adversarial example
                    adversarial_attack = Attack_methods('fgsm', self.model, X , y, self.epsilon, self.alpha, self.iter_num)
                    delta = adversarial_attack.choose_method()
    
                    # Update network parameters
                    output = self.model(torch.clamp(X + delta, 0, 1))
                    loss = nn.CrossEntropyLoss()(output, y)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
    
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
    
            train_time = time.time()
            self.logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f', epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
            torch.save(self.model.state_dict(), self.fname)



