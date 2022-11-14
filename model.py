import timm
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchinfo import summary


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p).view(x.shape[0], -1)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class APTOSModel(nn.Module):
    def __init__(self):
        super(APTOSModel, self).__init__()
        self.model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=1)
        self.model.global_pool = GeM()

    def forward(self, img):
        return self.model(img)


if __name__ == '__main__':
    model = APTOSModel()
    # img = torch.ones(1, 3, 224, 224)
    # out = model(img)
    # print(out.shape)
    summary(model, input_size=(16, 3, 224, 224))
