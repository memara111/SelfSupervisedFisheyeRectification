import torch
import torch.nn as nn
import torchvision.transforms as transforms


def load_vgg(pretrained=False, name="vgg11", hub_ref="pytorch/vision:v0.6.0"):
    """``torch.hub`` is the historical way this model was built, but it makes model
    *construction* need GitHub access. Fall back to the local torchvision copy.
    """
    try:
        return torch.hub.load(hub_ref, name, pretrained=pretrained)
    except Exception as error:  # offline, proxy, unpinned ref, ...
        print("WARNING: torch.hub.load({!r}, {!r}) failed ({}); "
              "falling back to torchvision.models.".format(hub_ref, name, error))
        factory = getattr(__import__("torchvision.models", fromlist=[name]), name)
        return factory(weights="DEFAULT" if pretrained else None)


class ParametersEstimationModule(nn.Module):
    """Encoder/decoder backbone feeding a VGG head that regresses one scalar: the
    distortion parameter of the division model.
    """

    def __init__(self, in_channels=3, vgg_pretrained=False, freeze_vgg=False,
                 output_range=(-1.0, 1.0), vgg_name="vgg11"):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1),
            nn.Conv2d(64, 64, 3, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1),
            nn.Conv2d(128, 128, 3, 1),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(64, 3, 2, 2),
        )
        self.vgg = load_vgg(pretrained=vgg_pretrained, name=vgg_name)
        self.vgg.classifier[6] = nn.Linear(self.vgg.classifier[6].in_features, 1)
        self.sigmoid = nn.Sigmoid()

        self.freeze_vgg = bool(freeze_vgg)
        if self.freeze_vgg:
            for param in self.vgg.parameters():
                param.requires_grad_(False)

        # The raw head used to be returned unbounded, so a prediction of e.g. -7 was
        # fed straight into the division model (sqrt of a negative number -> NaN loss)
        # and clamped only later by the renderer. Bound it to the model's real range.
        self.low, self.high = float(output_range[0]), float(output_range[1])

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.vgg(x)
        x = torch.flatten(x)
        return self.low + (self.high - self.low) * self.sigmoid(x)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vgg:
            # keep dropout / any running statistics of the frozen branch inert
            self.vgg.eval()
        return self

    def getTransforms(self):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    @staticmethod
    def denormalize(tensor):
        """Inverse of :meth:`getTransforms`, clamped into the PIL range.

        ``Normalize`` is ``(x - mean) / std``, so the inverse multiplies by ``std``
        *then* adds ``mean`` -- doing it in the other order is not the inverse.
        """
        mean = tensor.new_tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
        std = mean
        return (tensor * std + mean).clamp(0.0, 1.0)
