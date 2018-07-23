import numpy as np
from chainer import Chain
import chainer.functions as F

class DistillPredictor(Chain):
    def __init__(self, teacher, student, T = 1.0, alpha = 0.5):
        super(DistillPredictor, self).__init__(predictor=student)
        self.teacher = teacher
        self.T = T
        self.alpha = alpha
        
    def __call__(self, x, t):
        logits_s = self.predictor(x)
        logits_t = self.teacher(x)
        # hard target loss
        hard_loss = F.softmax_cross_entropy(logits_s,t)
        # soft target loss with temperature
        s_soft = logits_s / self.T
        t_soft = logits_t / self.T
        soft_loss = softloss(t_soft, s_soft)
        # total loss
        loss = hard_loss * (1 - self.alpha) + soft_loss * self.alpha * self.T * self.T
        return loss

    def predict(self, x):
        y = self.predictor(x)
        return F.softmax(y)

def softloss(t_soft, s_soft):
    n = t_soft.shape[0]
    return - F.sum(F.softmax(t_soft) *  F.log_softmax(s_soft)) / n