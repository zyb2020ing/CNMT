import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits).reshape(-1)#torch.Size([24084480])
        targets = targets.view(-1)#torch.Size([250880])
   

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice

def consistency_loss(pred, target):
    """
    计算预测与目标之间的一致性损失（MSE）。
    
    参数：
    - pred: 模型对无标签数据的预测 (softmax 输出)
    - target: 伪标签 (softmax 输出)

    返回：
    - loss: 一致性损失 (MSE)
    """
    return torch.mean((pred - target) ** 2)
