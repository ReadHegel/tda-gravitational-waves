from torch import nn
import torch
import torch.nn.functional as F


class Clasificator(nn.Module):
    def __init__(self, with_betti: bool = True):
        super(Clasificator, self).__init__()
        self.with_betti = with_betti

        self.signal_backbone = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=16, stride=1),  
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),      

            nn.Conv1d(64, 128, kernel_size=16, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),       

            nn.Conv1d(128, 256, kernel_size=16, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),       

            nn.Conv1d(256, 512, kernel_size=32, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4), 
        )

        self.betti_curve_backbone = nn.Sequential(
            *[
                nn.Conv1d(2, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
        )
        self.classification_head = nn.Sequential(
            *[
                nn.LazyLinear(out_features=128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
            ]
        )

    def forward(self, signal, betti_curve):
        signal_embedded = torch.flatten(
            self.signal_backbone(signal),
            start_dim=1,
        )

        if self.with_betti:
            betti_curve_embedded = torch.flatten(
                self.betti_curve_backbone(betti_curve),
                start_dim=1,
            )

            x = torch.concat(
                (signal_embedded, betti_curve_embedded),
                dim=1,
            )
        else: 
            x = signal_embedded

        return self.classification_head(x)


def train_epoch(model, device, train_loader, optim, epoch): 
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): 
        signal, betti_curve = data
        signal, betti_curve = signal.to(device), betti_curve.to(device)
        target = target.to(device)

        optim.zero_grad()

        logits = model(signal, betti_curve)
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optim.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test_epoch(model, device, test_loader): 
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader: 
            signal, betti_curve = data
            signal, betti_curve = signal.to(device), betti_curve.to(device)
            target = target.to(device)

            logits = model(signal, betti_curve)
            loss = F.cross_entropy(logits, target)

            test_loss += loss
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def train(model, device, train_loader, test_loader, optim, epochs): 
    model = model.to(device)
    for epoch in range(1, epochs+1): 
        train_epoch(model, device, train_loader, optim, epoch)
        test_epoch(model, device, test_loader)


