# import packages
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchsummary import summary

from sklearn.metrics import accuracy_score,classification_report

from tqdm import tqdm_notebook as tqdm
import time
import warnings
warnings.simplefilter("ignore")

from google.colab import drive 
drive.mount('/content/drive')


## Set up parameters
# DATASET INFO
NUM_CLASSES = 4 # set the number of classes in your dataset
training_dir = '/content/drive/MyDrive/ColabNotebooks/brain_tumor_classification/Training/'
testing_dir = '/content/drive/MyDrive/ColabNotebooks/brain_tumor_classification/Testing/'

# DATALOADER PROPERTIES
BATCH_SIZE = 4 
IMAGE_SIZE = (128, 128)

# LEARNING RATE PARAMETERS
epochs = 50

### GPU SETTINGS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


## set up data transformation functions and data loaders
def data_transforms(x):
  ##for the training images, do a resize random crop based on IMAGE_SIZE,
  ##random rotation, totensor, normalize
    if x == 'training':
        data_transformation = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE),
            transforms.RandomRotation(degrees=(-25,20)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
  ##for test data, resize and centercrop then totensor, normalize
    else:
        data_transformation=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    
    return data_transformation


### create training set, testing set, val set by the ImageFolder function
### split the testset into a test set and valset

train_dset = datasets.ImageFolder(training_dir, transform = data_transforms('training'))
test_dset = datasets.ImageFolder(testing_dir, transform = data_transforms('test'))
test_dset, val_dset = torch.utils.data.random_split(test_dset, [150, 244])


### wrap the datasets with an iterable using DataLoader

train_loader = DataLoader(train_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# visualize dataset
def matplotlib_imshow(img, one_channel=False):
    plt.figure(figsize=(10,10))
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg)
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(train_loader)
images, labels = dataiter.next()

# Create a grid from the images and show them
img_grid = torchvision.utils.make_grid(images)
matplotlib_imshow(img_grid, one_channel=True)


### TESTING LOOP
def test(model,testloader):
    with torch.no_grad():
        n_correct=0
        n_samples=0
        y_pred=[]
        y_actual=[]
        for i,(images,labels) in enumerate(testloader):
            images=images.to(device)
            labels=labels.to(device)
            
            outputs=model(images)
            
            y_actual+=list(np.array(labels.detach().to('cpu')).flatten())
        # value ,index
            _,predictes=torch.max(outputs,1)
            y_pred+=list(np.array(predictes.detach().to('cpu')).flatten())
        # number of samples in current batch
            n_samples+=labels.shape[0]

            n_correct+= (predictes==labels).sum().item()

        y_actual=np.array(y_actual).flatten()
        y_pred=np.array(y_pred).flatten()
        print(np.unique(y_pred))
        acc = classification_report(y_actual,y_pred,target_names=train_dset.classes)
        print(f"{acc}")


### TRAINING LOOP
def train(model,train_loader,criterion,optimizer,val_loader,epochs=25):    

    train_losses = []
    val_losses = []
    train_auc = []
    val_auc = []
    train_auc_epoch = []
    val_auc_epoch = []
    best_acc = 0.0
    min_loss = np.Inf

    since = time.time()
    y_actual=[]
    y_pred=[]
    for e in range(epochs):

      #scale LR parameter

        if best_acc > 0.4:
          learning_rate = 0.005

        if best_acc > 0.6:
          learning_rate = 0.001

        if best_acc > 0.72:
          learning_rate = 0.0005

        if best_acc > 0.80:
          learning_rate = 0.0001

        if best_acc > 0.85:
          learning_rate = 0.00005

        if best_acc > 0.885:
          learning_rate = 0.00001

        if best_acc > 0.9:
          learning_rate = 0.000001


        y_actual=[]
        y_pred=[]
        train_loss = 0.0
        val_loss = 0.0

        # Train the model
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader, total=int(len(train_loader)))):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Loss and accuracy
            train_loss += loss.item()
            
            _,predictes=torch.max(outputs,1)
            y_actual += list(labels.data.cpu().numpy().flatten()) 
            y_pred += list(predictes.detach().cpu().numpy().flatten())
        train_auc.append(accuracy_score(y_actual, y_pred))

        # Evaluate the model
        model.eval()
        for i, (images, labels) in enumerate(tqdm(val_loader, total=int(len(val_loader)))):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Loss and accuracy
            val_loss += loss.item()
            _,predictes=torch.max(outputs,1)
            y_actual += list(labels.data.cpu().numpy().flatten()) 
            y_pred += list(predictes.detach().cpu().numpy().flatten())
        
        val_auc.append(accuracy_score(y_actual, y_pred))

        # Average losses and accuracies
        train_loss = train_loss/len(train_loader)
        val_loss = val_loss/len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        training_auc = train_auc[-1]
        validation_auc = val_auc[-1]
        train_auc_epoch.append(training_auc)
        val_auc_epoch.append(validation_auc)

        # Updating best validation accuracy
        if best_acc < validation_auc:
            best_acc = validation_auc

        # Saving best model
        if min_loss >= val_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            min_loss = val_loss

        print('EPOCH {}/{} Train loss: {:.6f},Validation loss: {:.6f}, Train AUC: {:.4f}  Validation AUC: {:.4f}\n  '.format(e+1, epochs,train_loss,val_loss, training_auc,validation_auc))
        print('-' * 10)
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy: {:4f}'.format(best_acc))
    return train_losses,val_losses,train_auc,val_auc,train_auc_epoch,val_auc_epoch


## CREATE MOBILENET CLASS

class MyMobileNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.pretrained = torchvision.models.mobilenet_v3_large(pretrained=True)
    self.new_linear = nn.Linear(1000, 4)

  def forward(self, x):
    return self.new_linear(self.pretrained(x))


## RUN MODEL

model = MyMobileNet().to(device)

criterion = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.Adamax(model.parameters(),lr=learning_rate)

train_losses,val_losses,train_auc,val_auc,train_auc_epoch,val_auc_epoch=train(model,train_loader,criterion,optimizer,val_loader,80)


## PLOT DATA AND TEST DATA

plt.figure(figsize=(20,5))
plt.plot(train_auc_epoch, '-o', label="train")
plt.plot(val_auc_epoch, '-o', label="val")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over epoch - MobileNet Large")
plt.legend()

test(model,test_loader)
