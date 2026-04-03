import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from transformers import ViTForImageClassification
from torch.utils.data import DataLoader


def get_data(batch_size):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2761]),
    ])
    valset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return val_loader


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=100)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    val_loader = get_data(args.batch_size)

    model.eval()
    total = 0
    correct = 0
    loss = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(pixel_values=x)
            loss += criterion(out, y).item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

    print(f"Test loss: {loss/total:.4f} Test accuracy: {correct/total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    main(args)
