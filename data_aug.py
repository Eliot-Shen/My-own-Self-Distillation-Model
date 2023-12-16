import torchvision.transforms as transforms
dataset_transfrom = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                           [0.2023, 0.1994, 0.2010])
])