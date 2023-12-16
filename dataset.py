import torchvision
from torch.utils.tensorboard import SummaryWriter
import data_aug

# dataset_transfrom = torchvision.transforms.Compose([
#     torchvision.transforms.ToTensor()
# ])
train_set = torchvision.datasets.CIFAR100(root="./dataset", train=True, transform=data_aug.dataset_transfrom, download=False)
test_set = torchvision.datasets.CIFAR100(root="./dataset", train=False, transform=data_aug.dataset_transfrom, download=False)
if __name__=='__main__':
    print("train_set's length:",len(train_set))
    print("test_set's length:",len(test_set))
    print(train_set[7][1])
    print(train_set[5][1])
    # train_set (第几个数据)(0-图片tensor, 1-对应类别)
    print("dataset's shape:",train_set[0][0].shape)

    writer = SummaryWriter("Visualize Picture")
    # 命令行参数 tensorboard --logdir="Visualize Picture"
    for i in range(10):
        img, target = test_set[i]
        writer.add_image("test_set", img, i)
    writer.close()  # 关闭读写
