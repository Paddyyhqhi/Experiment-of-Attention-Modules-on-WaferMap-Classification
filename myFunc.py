import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale
from skimage.measure import label
import cv2
import random
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torchvision.models import ResNet34_Weights, ResNet50_Weights
import pandas as pd
import seaborn as sns
import time
from PIL import Image
import torch.nn.functional as F

def convert_gray_to_rgb(wafer_map):
    output = (wafer_map / np.max(wafer_map) * 255).astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)
class CustomMaskTransform:
    def __call__(self, img):
        img_array = np.array(img)
        return label(masked_img)
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.features = dataframe['waferMap'].values
        self.labels = dataframe['failureNum'].values
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Convert numpy array to PIL Image
        sample = Image.fromarray(sample.astype(np.uint8))

        if self.transform:
            sample = self.transform(sample)
            # np.moveaxis(sample.numpy(), 0, -1)

        return sample, label
def get_dataset(size, bs, seed, augment=True):
    if augment:
        dataset = pd.read_pickle("dataset_aug.pkl")
    else:
        dataset = pd.read_pickle("dataset.pkl")
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, random_state=seed)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    train_dataset = CustomDataset(dataset_train, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_dataset = CustomDataset(dataset_test, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
    return train_dataloader, test_dataloader


def training(train_dataloader, test_dataloader, model, epochs, learning_rate, device):
    num_epochs = epochs
    period = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for data, label in train_dataloader:
            data, target = data.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == target).sum().item()
            total_correct += correct
            total_samples += data.size(0)

        train_loss.append(total_loss / len(train_dataloader))
        train_acc.append(100 * total_correct / total_samples)

        if (epoch + 1) % period == 0 or (epoch == num_epochs - 1):
            print(f'Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_acc[-1]:.2f}%')

    end_time = time.time()

    execution_time = end_time - start_time
    print(f"exution time: {execution_time} s")

    return model, (train_acc,test_acc,train_loss,test_loss)

def evaluate(test_dataloader, model, history, device, name):
    train_acc,test_acc,train_loss,test_loss = history[0], history[1], history[2], history[3]
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_target = []
    all_pred = []

    for data, target in test_dataloader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == target).sum().item()
        total_correct += correct
        total_samples += data.size(0)
        all_target.extend(target.cpu().numpy())
        all_pred.extend(predicted.cpu().numpy())


    patterns = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
            "Loc", "Random", "Scratch", "Near-full"]
    f1_scores = f1_score(all_target, all_pred, labels=[0,1,2,3,4,5,6,7], average=None)
    precision_scores = precision_score(all_target, all_pred, labels=[0,1,2,3,4,5,6,7], average=None)
    recall_scores = recall_score(all_target, all_pred, labels=[0,1,2,3,4,5,6,7], average=None)

    # Print scores for each class
    for i, (f1, prec, rec) in enumerate(zip(f1_scores, precision_scores, recall_scores)):
        print(f'{patterns[i]:20}:F1 Score: {100 * f1:.3f}%, Precision: {100 * prec:.3f}%, Recall: {100 * rec:.3f}%')
    df = {'f1_scores':f1_scores, 'precision':precision_scores, 'recall':recall_scores}
    print("=======================================================")

    # Calculate and print overall scores
    f1_macro = np.mean(f1_scores)
    f1_micro = f1_score(all_target, all_pred, labels=[0,1,2,3,4,5,6,7], average='micro')

    print(f'Overall f1_macro: {100 * f1_macro:.3f}%, f1_micro: {100 * f1_micro:.3f}%')

    # Calculate and print Accuracy
    f1_scores = list(f1_scores)
    precision_scores = list(precision_scores)
    recall_scores = list(recall_scores)

    accuracy = 100 * total_correct / total_samples
    print(f'Accuracy: {accuracy:.3f}%')
    f1_scores.extend([f1_macro, f1_micro, accuracy])
    precision_scores.extend(['' for _ in range(3)])
    recall_scores.extend(['' for _ in range(3)])
    df = {'f1_scores':f1_scores, 'precision':precision_scores, 'recall':recall_scores}
    patterns.extend(["f1 macro", "f1 micro", "accuracy"])
    df = pd.DataFrame(df, index=patterns)
    df.to_csv("Exp_result.csv")

    # Plot Confusion Matrix
    patterns = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
            "Loc", "Random", "Scratch", "Near-full"]
    conf_matrix = confusion_matrix(all_target, all_pred)
    accuracy_matrix = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=patterns, yticklabels=patterns)
    plt.title("Confusion Matrix")
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def GradCam(model, pred, imgs, i):
    img = imgs[i][None,:,:,:]
    pred = model(img)[:, pred.item()]
    pred.backward()
    gradients = model.get_activations_gradient()
    activations = model.get_activations().detach()
    b, k, u, v = gradients.size()
    heatmap = F.relu((gradients.view(b, k, -1).mean(2).view(b, k, 1, 1) * activations).sum(1, keepdim=True))
    # heatmap = temp
    heatmap = F.upsample(heatmap, size=(img.shape[-1], img.shape[-2]), mode='bilinear', align_corners=False)
    map_min, map_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - map_min).div(map_max - map_min).data
    heatmap = heatmap.view(img.shape[-1], img.shape[-2], -1)
    h = np.uint8(255 * heatmap.cpu().numpy())
    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    img = imgs[i].permute(1, 2, 0)
    result = h * 0.4 + img.cpu().numpy()*255
    output = (result / np.max(result) * 255).astype(np.uint8)
    return output

def GradCampp(model, pred, imgs, i):
    img = imgs[i][None,:,:,:]
    pred = model(img)[:, pred.item()]
    pred.backward()
    gradients = model.get_activations_gradient()
    activations = model.get_activations().detach()
    b, k, u, v = gradients.size()
    alpha_num = gradients.pow(2)
    alpha_denom = gradients.pow(2).mul(2) + \
            activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

    alpha = alpha_num.div(alpha_denom+1e-7)
    positive_gradients = F.relu(pred.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
    weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)
    heatmap = F.relu((weights * activations).sum(1, keepdim=True))
    heatmap = F.upsample(heatmap, size=(img.shape[-1], img.shape[-2]), mode='bilinear', align_corners=False)
    map_min, map_max = heatmap.min(), heatmap.max()
    heatmap = (heatmap - map_min).div(map_max - map_min).data
    heatmap = heatmap.view(img.shape[-1], img.shape[-2], -1)
    h = np.uint8(255 * heatmap.cpu().numpy())
    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    img = imgs[i].permute(1, 2, 0)
    result = h * 0.4 + img.cpu().numpy()*255
    output = (result / np.max(result) * 255).astype(np.uint8)
    return output

def show_GradCam(test, model, device):
    model.eval()
    patterns = ["Center", "Donut", "Edge-Loc", "Edge-Ring",
                "Loc", "Random", "Scratch", "Near-full"]
    collection = dict()
    for imgs, labels in test:
        imgs, labels = imgs.to(device), labels.to(device)
        for i in range(imgs.shape[0]):
            img = imgs[i][None,:,:,:]
            pred = model(img).argmax(dim=1)
            pred_label = patterns[pred.item()]
            if (pred_label not in collection) and (pred.item() == labels[i].item()):
                # build heatmap

                # results
                collection[pred_label] = [GradCam(model, pred, imgs, i), GradCampp(model, pred, imgs, i)]
        if len(collection) == len(patterns):
            break
    fig, ax = plt.subplots(8, 2, figsize=(10, 40))
    ax = ax.flatten()
    count = 0
    for label in collection.keys():
        ax[count].set_xticks([]), ax[count].set_yticks([])
        ax[count].set_title(label)
        ax[count].imshow(collection[label][0])
        count += 1
        ax[count].set_xticks([]), ax[count].set_yticks([])
        ax[count].set_title(label)
        ax[count].imshow(collection[label][1])
        count += 1
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()
    

    
if __name__ == "__main__":
    size = 224
    bs = 16
    seed = 60
    device = torch.device("cuda")
    same_seeds(seed)
    
    train, test = get_dataset(size, bs, seed)
    ## Make DANet
    # in_channels = 512
    # out_channels = in_channels // 4
    # model = DANet(in_channels, out_channels)
    # model.to(device)
    model = ModelName()
    model.to(device)
    epochs = 10
    learning_rate = 1e-4
    model, history = training(train, test, model, epochs, learning_rate, device)
    evaluate(test, model, history, device, "Model Name")
    
    show_GradCam(test, model, device)
