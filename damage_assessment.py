import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import glob
from torchvision.transforms import CenterCrop
import time
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np 
from collections import OrderedDict
from matplotlib.colors import ListedColormap



#Unet Model 

class Unet(nn.Module):

    def __init__(self, in_channels=3, out_channels_s=2, out_channels_c=5, init_features=16):
        super(Unet, self).__init__()

        features = init_features

        # Unet layers
        self.encoder1 = Unet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Unet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Unet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Unet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Unet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = Unet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = Unet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = Unet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = Unet._block(features * 2, features, name="dec1")

        self.conv_s = nn.Conv2d(in_channels=features, out_channels=out_channels_s, kernel_size=1)

        # Siamese classifier layers
        self.upconv4_c = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.conv4_c = Unet._block(features * 16, features * 16, name="conv4")

        self.upconv3_c = nn.ConvTranspose2d(features * 16, features * 4, kernel_size=2, stride=2)
        self.conv3_c = Unet._block(features * 8, features * 8, name="conv3")

        self.upconv2_c = nn.ConvTranspose2d(features * 8, features * 2, kernel_size=2, stride=2)
        self.conv2_c = Unet._block(features * 4, features * 4, name="conv2")

        self.upconv1_c = nn.ConvTranspose2d(features * 4, features, kernel_size=2, stride=2)
        self.conv1_c = Unet._block(features * 2, features * 2, name="conv1")

        self.conv_c = nn.Conv2d(in_channels=features * 2, out_channels=out_channels_c, kernel_size=1)


    def forward(self, x1, x2):

        # Unet on x1
        enc1_1 = self.encoder1(x1)
        enc2_1 = self.encoder2(self.pool1(enc1_1))
        enc3_1 = self.encoder3(self.pool2(enc2_1))
        enc4_1 = self.encoder4(self.pool3(enc3_1))

        bottleneck_1 = self.bottleneck(self.pool4(enc4_1))

        dec4_1 = self.upconv4(bottleneck_1)
        dec4_1 = torch.cat((dec4_1, enc4_1), dim=1)
        dec4_1 = self.decoder4(dec4_1)
        dec3_1 = self.upconv3(dec4_1)
        dec3_1 = torch.cat((dec3_1, enc3_1), dim=1)
        dec3_1 = self.decoder3(dec3_1)
        dec2_1 = self.upconv2(dec3_1)
        dec2_1 = torch.cat((dec2_1, enc2_1), dim=1)
        dec2_1 = self.decoder2(dec2_1)
        dec1_1 = self.upconv1(dec2_1)
        dec1_1 = torch.cat((dec1_1, enc1_1), dim=1)
        dec1_1 = self.decoder1(dec1_1)

        # Unet on x2
        enc1_2 = self.encoder1(x2)
        enc2_2 = self.encoder2(self.pool1(enc1_2))
        enc3_2 = self.encoder3(self.pool2(enc2_2))
        enc4_2 = self.encoder4(self.pool3(enc3_2))

        bottleneck_2 = self.bottleneck(self.pool4(enc4_2))

        dec4_2 = self.upconv4(bottleneck_2)
        dec4_2 = torch.cat((dec4_2, enc4_2), dim=1)
        dec4_2 = self.decoder4(dec4_2)
        dec3_2 = self.upconv3(dec4_2)
        dec3_2 = torch.cat((dec3_2, enc3_2), dim=1)
        dec3_2 = self.decoder3(dec3_2)
        dec2_2 = self.upconv2(dec3_2)
        dec2_2 = torch.cat((dec2_2, enc2_2), dim=1)
        dec2_2 = self.decoder2(dec2_2)
        dec1_2 = self.upconv1(dec2_2)
        dec1_2 = torch.cat((dec1_2, enc1_2), dim=1)
        dec1_2 = self.decoder1(dec1_2)

        # Siamese
        dec1_c = bottleneck_2 - bottleneck_1

        dec1_c = self.upconv4_c(dec1_c) # features * 16 -> features * 8
        diff_2 = enc4_2 - enc4_1 # features * 16 -> features * 8
        dec2_c = torch.cat((diff_2, dec1_c), dim=1) #512
        dec2_c = self.conv4_c(dec2_c)

        dec2_c = self.upconv3_c(dec2_c) # 512->256
        diff_3 = enc3_2 - enc3_1
        dec3_c = torch.cat((diff_3, dec2_c), dim=1) # ->512
        dec3_c = self.conv3_c(dec3_c)

        dec3_c = self.upconv2_c(dec3_c) #512->256
        diff_4 = enc2_2 - enc2_1
        dec4_c = torch.cat((diff_4, dec3_c), dim=1) #
        dec4_c = self.conv2_c(dec4_c)

        dec4_c = self.upconv1_c(dec4_c)
        diff_5 = enc1_2 - enc1_1
        dec5_c = torch.cat((diff_5, dec4_c), dim=1)
        dec5_c = self.conv1_c(dec5_c)

        return self.conv_s(dec1_1), self.conv_s(dec1_2), self.conv_c(dec5_c)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
        
        
#Intializing Data Loader 
class CustomDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        """
        Initializes the dataset.
        
        Args:
            base_dir (str): Base directory containing the images and masks.
            transform (callable, optional): Transform to apply to images and masks.
        """
        self.image_paths = sorted(glob.glob(os.path.join(base_dir, "**/images/*_pre_disaster.png"), recursive=True))
        self.transform = transform

    def __len__(self):
        """
        Returns the number of items in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Gets the pre-disaster image, post-disaster image, and corresponding mask.
        
        Args:
            idx (int): Index of the dataset item.
        
        Returns:
            tuple: (pre_image, post_image, mask)
        """
        # Get paths for pre-disaster and post-disaster images
        pre_image_path = self.image_paths[idx]
        post_image_path = pre_image_path.replace("_pre_disaster", "_post_disaster")

        # Load images
        pre_image = Image.open(pre_image_path).convert("RGB")
        post_image = Image.open(post_image_path).convert("RGB")

        # # Get path for the mask
        # pre_mask_path = pre_image_path.replace("images", "masks")
        # pre_mask = Image.open(pre_mask_path).convert("L")
        
        post_mask_path = post_image_path.replace("images", "masks")
        post_mask = Image.open(post_mask_path).convert("L")

        # Apply transformations if specified
        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)
            # pre_mask = transforms.ToTensor()(pre_mask).squeeze(0).long()
            post_mask = transforms.ToTensor()(post_mask).squeeze(0).long()


        return pre_image, post_image, post_mask


def DamageDataLoader(dataset_path,image_size,batch_size,train=False):

        
        #setting image transformation
        image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        ])
         
        # Load dataset
        dataset = CustomDataset(dataset_path, image_transform)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader 

def compute_iou(pred, target, num_classes):
    iou_list = []
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()
    
    for cls in range(num_classes):
        intersection = np.logical_and(pred == cls, target == cls).sum()
        union = np.logical_or(pred == cls, target == cls).sum()
        if union == 0:
            iou_list.append(np.nan)  # Ignore NaNs for classes not present
        else:
            iou_list.append(intersection / union)

    return np.nanmean(iou_list)  # Average IoU over all classes

# Define F1-Score computation function
def compute_f1_score(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()
    f1_scores = []

    for cls in range(num_classes):
        tp = np.logical_and(pred == cls, target == cls).sum()
        fp = np.logical_and(pred == cls, target != cls).sum()
        fn = np.logical_and(pred != cls, target == cls).sum()

        if tp + fp + fn == 0:
            f1_scores.append(np.nan)  # Ignore NaNs for classes not present
        else:
            f1_scores.append(2 * tp / (2 * tp + fp + fn))

    return np.nanmean(f1_scores)  # Average F1-Score over all classes

#Training, Testing
def DamageAssessment(dataset_path):
    
    device = torch.device("cpu")
    learning_rate = 0.001 
    num_epochs = 15
    batch_size = 16
    image_size = 256 
    
    model = Unet().to(device)
    
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #train and test data loaders 
    trainloader, testloader = DamageDataLoader(dataset_path,image_size,batch_size,True)
 
    #Model Training 
    model.train()
    
    train_losses, test_losses = [], []
    train_ious, test_ious = [], []
    train_f1s, test_f1s = [], []    
    
    num_classes = 5
    
    print("Starting Training Loop...")
    
    start_train_time = time.time() 
    
    for epoch in range(num_epochs):
        
        train_loss, train_iou, train_f1 = 0, 0, 0
        model.train()

        
        for pre_img, post_img, post_mask in tqdm(trainloader):
            pre_img = pre_img.to(device)
            post_img = post_img.to(device)
            post_mask = post_mask.to(device)
            optimizer.zero_grad()
            
            outputs = model(pre_img,post_img)
            resized_mask = F.interpolate(post_mask.unsqueeze(1).float(),  size=(256, 256), mode='nearest').squeeze(1).long()
            loss = criterion(outputs[1],resized_mask)
            
        
            #Backprop 
            loss.backward() 
            optimizer.step() 
            optimizer.zero_grad()
            
            train_loss +=loss.item()
            train_iou += compute_iou(outputs[1], resized_mask, num_classes)
            train_f1 += compute_f1_score(outputs[1], resized_mask, num_classes)
            
        train_losses.append(train_loss / len(trainloader))
        train_ious.append(train_iou / len(trainloader))
        train_f1s.append(train_f1 / len(trainloader))
            
        
        print("Total Time Taken For Training In Seconds --> ", time.time()-start_train_time)
        
        
        #Testing 
        model.eval() 
        test_loss, test_iou, test_f1 = 0, 0, 0
        
        print("Starting Testing Loop...")

        with torch.no_grad(): 
            for pre_img_test, post_img_test, mask_test in tqdm(testloader): 
                
                pre_img_test, post_img_test, mask_test = pre_img_test.to(device), post_img_test.to(device), mask_test.to(device)
                output_test = model(pre_img_test, post_img_test)
                resized_post_mask = F.interpolate(mask_test.unsqueeze(1).float(), size=(256, 256), mode='nearest').squeeze(1).long()
                            
                loss = criterion(output_test[1],resized_post_mask)

                test_loss += loss.item()
                test_iou += compute_iou(output_test[1], resized_post_mask, num_classes)
                test_f1 += compute_f1_score(output_test[1], resized_post_mask, num_classes)  
            
        test_losses.append(test_loss / len(testloader))
        test_ious.append(test_iou / len(testloader))
        test_f1s.append(test_f1 / len(testloader))
        
    
    #plotting the performance metrics 
    # Plot metrics
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(range(len(train_losses)), train_losses, marker='o', linestyle='-', label='Train Loss', color='blue')
    plt.plot(range(len(test_losses)), test_losses, marker='o', linestyle='-', label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()

    # IoU plot
    plt.subplot(1, 3, 2)
    plt.plot(range(len(train_ious)), train_ious, marker='o', linestyle='-', label='Train IoU', color='blue')
    plt.plot(range(len(test_ious)), test_ious, marker='o', linestyle='-', label='Validation IoU', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title('Training vs Validation IoU')
    plt.legend()

    # F1-Score plot
    plt.subplot(1, 3, 3)
    plt.plot(range(len(train_f1s)), train_f1s, marker='o', linestyle='-', label='Train F1-Score', color='blue')
    plt.plot(range(len(test_f1s)), test_f1s, marker='o', linestyle='-', label='Validation F1-Score', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.title('Training vs Validation F1-Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

    
    dataset = CustomDataset(args.input, transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]))
    visualize_predictions(model, dataset, device)
    

    
def visualize_predictions(model, dataset, device):
    
    model.eval()
    
    sample_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    pre_image, post_image, post_mask = next(iter(sample_loader))
    pre_image, post_image = pre_image.to(device), post_image.to(device)

    with torch.no_grad():
        output = model(pre_image, post_image)

    # Apply softmax and get predicted class labels
    softmax = torch.nn.Softmax(dim=1)
    preds_seg = torch.argmax(softmax(output[1]), dim=1)
    preds_cls = torch.argmax(softmax(output[2]), dim=1)
    
    print("tensors for classficiation: ", torch.unique(preds_cls))
    print("\n tensors for segmetations ", torch.unique(preds_seg))

    colors = ['black', 'green', 'yellow', 'orange', 'red']
    cmap = ListedColormap(colors)

    # Visualization
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    ax[0].imshow(np.squeeze(pre_image.cpu().squeeze(0).permute(1, 2, 0).numpy()))
    ax[0].set_title('Pre-Disaster Image')

    ax[1].imshow(np.squeeze(post_image.cpu().squeeze(0).permute(1, 2, 0).numpy()))
    ax[1].set_title('Post-Disaster Image')

    ax[2].imshow(np.squeeze(preds_seg.cpu().numpy()), cmap=cmap, interpolation='nearest')
    ax[2].set_title('Segmentated Output')

    ax[3].imshow(np.squeeze(preds_cls.cpu().numpy()), cmap=cmap, interpolation='nearest')
    ax[3].set_title('Classified Output')

    plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train disaster assessment model.")
    parser.add_argument('--input', required=True, help='Path to dataset directory')
    args = parser.parse_args()

    DamageAssessment(args.input)
    