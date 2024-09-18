import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, embed_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

class RegressionHead(nn.Module):
    def __init__(self, embed_size, output_size):
        super(RegressionHead, self).__init__()
        self.fc = nn.Linear(embed_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

class OutputLayer(nn.Module):
    def __init__(self, embed_size, num_classes=None, output_size=None):
        super(OutputLayer, self).__init__()
        if num_classes is not None:
            self.head = ClassificationHead(embed_size, num_classes)
        elif output_size is not None:
            self.head = RegressionHead(embed_size, output_size)
        else:
            raise ValueError("Either num_classes or output_size must be provided")

    def forward(self, x):
        return self.head(x)

