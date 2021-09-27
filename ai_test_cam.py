import cv2
import glob
from torchvision import transforms
from torch.autograd import Variable
import torch
from PIL import Image
import numpy as np
import time

capture = cv2.VideoCapture(1)

map_location = torch.device('cuda')
device = map_location
model = torch.load('data_aug.pth')
model.to(map_location)
model.eval()

classes = ['background', 'organic', 'paper', 'plastic']

loader = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def image_loader(image_name):
    global device
    image = Image.fromarray(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device)


def take_photo(frame):
    frame = frame[:, 85:420]
    # frame = np.multiply(frame,0.8)
    cv2.imwrite('imagen.jpg', cv2.resize(frame, (224, 224)))
    time.sleep(1)
    img = cv2.imread('imagen.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image_loader(img)
    outputs = model(image)
    ps = torch.exp(outputs)
    # print(ps.tolist()[0])
    _, predicted = torch.max(ps.data, 1)
    predicted_class = classes[predicted[0]]
    # print(predicted_class)
    return ps.tolist()[0], predicted_class


while 1:
    ret, frame = capture.read()
    cv2.imshow('pulsa C para capturar', frame)
    k = cv2.waitKey(100)
    # print(k)
    if k == 27:  # Esc
        break
    if k == 99:  # C
        attemps = 0
        while attemps <= 3:
            predictions, class_predicted = take_photo(frame)
            print(f'\nPredictions {predictions}')
            if max(predictions) > 0.85:
                print('Predicted class:', class_predicted)
                break
            else:
                if attemps < 3:
                    print('Probabilities are not enough to give a right answer, trying again...')
                else:
                    print('Could not give a correct answer, the best prediction was:', class_predicted)
                attemps += 1

capture.release()
cv2.destroyAllWindows
