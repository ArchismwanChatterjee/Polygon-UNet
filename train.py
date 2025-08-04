import os
import json
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from model import UNetConditional


class PolygonDataset(Dataset):
    def __init__(self, data_json, input_dir, output_dir, color_to_idx, transform=None):
        with open(data_json, 'r') as f:
            self.data = json.load(f)
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.color_to_idx = color_to_idx
        self.transform = transform

        self.samples = []
        for item in self.data: 
            input_img = item['input_polygon'] 
            color = item['colour']
            output_img = item['output_image']
            self.samples.append((input_img, color, output_img))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_img_name, color_name, output_img_name = self.samples[idx]

        input_img = Image.open(os.path.join(self.input_dir, input_img_name)).convert("L")
        output_img = Image.open(os.path.join(self.output_dir, output_img_name)).convert("RGB")

        if self.transform:
            input_img = self.transform(input_img)
            output_img = self.transform(output_img)

        color_idx = self.color_to_idx[color_name]
        color_onehot = torch.zeros(len(self.color_to_idx))
        color_onehot[color_idx] = 1.0

        return input_img, color_onehot, output_img

def train_model():
    wandb.init(project="polygon-coloring-unet")

    train_input_dir = "/content/drive/MyDrive/dataset/training/inputs"
    train_output_dir = "/content/drive/MyDrive/dataset/training/outputs"
    train_json = "/content/drive/MyDrive/dataset/training/data.json"

    val_input_dir = "/content/drive/MyDrive/dataset/validation/inputs"
    val_output_dir = "/content/drive/MyDrive/dataset/validation/outputs"
    val_json = "/content/drive/MyDrive/dataset/validation/data.json"


    colors = ["cyan","red", "green", "blue", "yellow", "purple","orange","magenta"]
    color_to_idx = {c: i for i, c in enumerate(colors)}

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])


    train_dataset = PolygonDataset(train_json, train_input_dir, train_output_dir, color_to_idx, transform)
    val_dataset = PolygonDataset(val_json, val_input_dir, val_output_dir, color_to_idx, transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


    model = UNetConditional(in_channels=1, out_channels=3, embed_dim=len(colors)).cuda()


    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, color_onehot, targets in train_loader:
            inputs, color_onehot, targets = inputs.cuda(), color_onehot.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs, color_onehot)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, color_onehot, targets in val_loader:
                inputs, color_onehot, targets = inputs.cuda(), color_onehot.cuda(), targets.cuda()
                outputs = model(inputs, color_onehot)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


    torch.save(model.state_dict(), "unet_polygon_color.pth")
    wandb.save("unet_polygon_color.pth")


if __name__ == "__main__":
    train_model()
