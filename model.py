import timm
import torch.nn.functional as F
import torch.nn as nn
import torch


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class APTOSModel(nn.Module):
    def __init__(self):
        super(APTOSModel, self).__init__()
        self.backbone = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=5)
        self.pooling = GeM()
        self.linear_fn = nn.Linear(self.backbone.classif.in_features, 5)

    def forward(self, img):
        out = self.backbone.forward_features(img)
        out = self.pooling(out)
        out = self.linear_fn(out.view(out.size(0), -1))
        return out


if __name__ == '__main__':
    model = APTOSModel()
    img = torch.ones(1, 3, 224, 224)
    out = model(img)
    print(out.shape)