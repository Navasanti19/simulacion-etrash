import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import seaborn as sb
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2
import glob
import pathlib


def data_aug(path: str = 'base/'):
    factors = [0.8, 1.2]
    folders_paths = [path + 'train/organic/', path + 'train/background/',
                     path + 'train/plastic/', path + 'train/paper/']

    for dir in folders_paths:
        print(dir)
        folder = dir
        files_number = 0
        files = os.listdir(dir)

        for index, file in enumerate(files):
            # print(index, file)
            try:
                os.rename(os.path.join(dir, file), os.path.join(dir, '_'.join([str(index + 1), '.jpg'])))
            except:
                pass
        n_files = len(glob.glob(folder + "*.jpg"))
        for i in range(n_files):
            cv2.imwrite(folder + str(i + 1) + '_m.jpg', cv2.flip(cv2.imread(folder + str(i + 1) + '_.jpg'), 1))
            for x in range(len(factors)):
                frame1 = np.multiply(cv2.imread(folder + str(i + 1) + '_.jpg'), factors[x])
                cv2.imwrite(folder + str(i + 1) + '_' + str(factors[x]) + '.jpg', frame1)


def model_training(dir: str = 'base/'):
    train_dir = 'train'
    valid_dir = 'test'

    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    model = models.resnet18(pretrained=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(p=0.6),
        nn.Linear(256, 4),
        nn.LogSoftmax(dim=1))
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    epochs = 5
    criterion = nn.NLLLoss()
    model.to(device)
    running_loss = 0
    steps = 0
    max_accuracy = 0

    print('Training Started!')

    for e in range(epochs):

        print('Epoch number: ', e + 1)

        for inputs, labels in train_loader:

            # Training Loop

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            steps += 1

            if steps == 5:
                model.eval()
                accuracy = 0
                valid_loss = 0

                with torch.no_grad():

                    for inputs, labels in valid_loader:
                        # Validation Loop

                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        ps = torch.exp(outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        loss_valid = criterion(outputs, labels)
                        valid_loss += loss_valid.item()

                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    if accuracy / len(valid_loader) > max_accuracy:
                        max_accuracy = accuracy / len(valid_loader)
                        torch.save(model, 'pipeline_model.pth')

                    print(
                        f"Train loss: {running_loss / steps:.3f}.. "
                        f"Validation loss: {valid_loss / len(valid_loader):.3f}.. "
                        f"Validation accuracy: {accuracy / len(valid_loader):.3f}")

                running_loss = 0
                steps = 0
                model.train()

    print('Training finished!')
    return model


# IMSHOW FOR SANITY CHECK

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def sanity_check(model, dir: str = 'base/'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ['background', 'organic', 'paper', 'plastic']

    image = Image.open(base + 'test/background/80.jpg')
    preprocess = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    image = preprocess(image)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 2, 1)
    imshow(image, ax);
    model.eval();
    model.to(device)

    image = image.unsqueeze(0)
    inputs = image.to(device)
    output = model.forward(inputs)
    ps = torch.exp(output)
    prob = torch.topk(ps, 3)[0].tolist()[0]
    index = torch.topk(ps, 3)[1].tolist()[0]
    print(prob)
    resindex = []
    for i in index:
        resindex.append(classes[i])
    print(resindex)
    print()
    plt.subplot(2, 1, 2)
    sb.barplot(x=resindex, y=prob, palette='Reds')
    sb.set_style("white")
    plt.xlabel("Prediction Probabilities")
    plt.ylabel("")
    plt.show();
    # torch.save(model, 'data_aug2.pth')


def pipeline(path: str = 'base/', factors: list = [0.8, 1.2]):
    # DATA AUGMENTATION

    data_aug(path)

    # MODEL TRAINING

    pipeline_model = model_training(path)

    # SANITY CHECK FOR IMAGE PREDICTION PREVIEW

    sanity_check(pipeline_model, path)


if __name__ == '__main__':
    pipeline()
