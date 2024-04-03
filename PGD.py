import torch
import torch.nn as nn

from root_class_attack import Attack

# FGM/PGD attack
class FGM(Attack):

    def __init__(self, model, norm='inf', targeted=False, img_range=(0, 1)):
        super().__init__(model, targeted=targeted, img_range=img_range)
        self.loss_fn = nn.CrossEntropyLoss()
        self.norm = norm

    def __call__(self, x, y, eps=0.0):

        x = x.to(self.device).detach()
        y = y.to(self.device).detach()
        x.requires_grad = True

        outs = self.model(x)
        self.model.zero_grad()
        loss = self.loss_fn(outs, y)
        loss.backward()

        if self.norm == 'inf':
            adv = x + (1 - 2*self.targeted) * eps * x.grad.sign()
        else:
            tmp = x.grad.view(x.size(0), -1).norm(self.norm, dim=-1)
            tmp = tmp.view(-1, 1, 1, 1)
            adv = x + (1 - 2*self.targeted) * eps * x.grad / tmp

        return torch.clamp(adv, *self.img_range).detach()

class PGD(FGM):

    def __init__(self, model, rand_restarts=0, init_radius=0, iters=10,
                 norm='inf', targeted=False, img_range=(0,1)):
        super().__init__(model, norm=norm, targeted=targeted, img_range=img_range)
        self.init_radius = init_radius
        self.rand_restarts = rand_restarts
        self.iters = iters

    def __call__(self, x, y, eps=0.0):

        x = x.to(self.device)
        y = y.to(self.device)
        alpha = eps / self.iters
        loss_fn_no_reduction = nn.CrossEntropyLoss(reduction='none')
        targT = (torch.ones_like(y)*self.targeted).bool()
        b_size = x.shape[0]
        best_adv = x.clone()
        best_loss = [float('inf') if self.targeted else 0] * x.shape[0]

        for i in range(self.rand_restarts + 1):
            if self.rand_restarts:
                x_ = x + torch.zeros_like(x).uniform_(-self.init_radius, self.init_radius)
            else:
                x_ = x

            for j in range(self.iters):
                x_ = super().__call__(x_, y, alpha)

                if self.norm == 'inf':
                    x_ = torch.max(torch.min(x_, x + eps), x - eps)
                else:
                    red_range = list(range(1, len(x.shape)))
                    pert = x_ - x
                    scale = torch.sum(pert ** 2, dim=red_range, keepdim=True)
                    scale[(scale.clone() == 0)] = 1e-9
                    factor = torch.min(torch.ones((1), device=self.device), eps / scale)
                    x_ = x + pert * factor

            logits = self.model(x_)
            predictions = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            success = torch.logical_xor((predictions == y), targT)
            losses = loss_fn_no_reduction(logits, y)

            for k, (s, l) in enumerate(zip(success, losses)):
                tmp = (1 - 2 * self.targeted)
                if (s and tmp * l >= tmp * best_loss[k]) or i == 0:
                    best_adv[k] = x_[k].clone()
                    best_loss[k] = l

        return best_adv

