import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision import datasets, transforms
import net
import time

import json
# import net_forward

# import Websocket.iwebsocket as iwebsocket


# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# device = "cuda:6" if torch.cuda.is_available() else "cpu"


def arg_parse():
    parser = argparse.ArgumentParser(
        description='choose by yourself.')
    parser.add_argument('--Dataset',
                        type=str,
                        default='imagenet_100',
                        choices=['MNIST', 'CIFAR10', 'ImageNet', 'imagenet_100', 'Military'],
                        help='type of dataset')
    parser.add_argument('--Model',
                        type=str,
                        default='AlexNet',
                        choices=['LeNet','AlexNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', "GoogleNet"],
                        help='type of net')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='batch size of dataset')
    parser.add_argument('--epoch',
                        type=int,
                        default=20,
                        help='batch size of distilled data')
    parser.add_argument('--json',
                        type=bool,
                        default=False,
                        help='create json')
    parser.add_argument('--GPU',
                        type=int,
                        default=6,
                        choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='type of dataset')

    args = parser.parse_args()
    return args


args = arg_parse()
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def Train(Dataset,Model,ws_client):
def Train(Dataset, Model):
    # 对于训练的时候输入有哪些变量控制，数据集和模型
    # Load the dataset and model
    print(Dataset, Model)

    Dataset_dir = '/data/Newdisk/zhaoxiaoming/datasets'

    if Dataset == 'MNIST':

        if Model == 'AlexNet':
            model = net.alexnet()
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=2,
                                          padding=2)
            model.classifier[6] = nn.Linear(4096, 10)
            model = model.to(device)

        if Model == 'VGG16':
            model = net.VGG16()
            model.classifier[6] = nn.Linear(4096, 10)
            model = model.to(device)

        if Model == 'VGG19':
            model = net.VGG19()
            model.classifier[6] = nn.Linear(4096, 10)
            model = model.to(device)

        transform_mnist = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor()
        ])

        trainset = torchvision.datasets.ImageFolder(root=os.path.join(Dataset_dir, Dataset, 'train'),
                                                    transform=transform_mnist)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(Dataset_dir, Dataset, 'test'),
                                                   transform=transform_mnist)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if Dataset == 'CIFAR10':

        if Model == 'AlexNet':
            model = net.alexnet_cifar10()
            model = model.to(device)

        if Model == 'VGG16':
            model = net.VGG16()
            model.classifier[6] = nn.Linear(4096, 10)
            model = model.to(device)

        if Model == 'VGG19':
            model = net.VGG19()
            model.classifier[6] = nn.Linear(4096, 10)
            model = model.to(device)

        transform_cifar10 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = torchvision.datasets.ImageFolder(root=os.path.join(Dataset_dir, Dataset, 'train'),
                                                    transform=transform_cifar10)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(Dataset_dir, Dataset, 'test'),
                                                   transform=transform_cifar10)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if Dataset == 'ImageNet':

        if Model == 'AlexNet':
            model = net.alexnet_cifar10()
            model.classifier[6] = nn.Linear(4096, 1000)
            model = model.to(device)

        if Model == 'VGG16':
            model = net.VGG16()
            model = model.to(device)

        if Model == 'VGG19':
            model = net.VGG19()
            model = model.to(device)

        if Model == 'GoogleNet':
            model = net.googlenet()
            print(model)
            #            model.fc = nn.Linear(1024, 10)
            model = model.to(device)
        transform_imagenet = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = torchvision.datasets.ImageFolder(root=os.path.join(Dataset_dir, Dataset, 'train'),
                                                    transform=transform_imagenet)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(Dataset_dir, Dataset, 'test'),
                                                   transform=transform_imagenet)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if Dataset == 'imagenet_100':

        lable_txt = '/data0/BigPlatform/ZJPlatform/000_Image/000-Dataset/imagenet100_Chinese.xlsx'

        if Model == 'LeNet':
            output = 100
            model = net.lenet()
            model.fc3 = nn.Linear(84, output)
            model = model.to(device)

        transform_imagenet = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if Model == 'AlexNet':
            model = net.alexnet()
            model.classifier[6] = nn.Linear(4096, 100)
            model = model.to(device)

        transform_google = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if Model == 'VGG11':
            model = net.VGG11()
            model.classifier[6] = nn.Linear(4096, 100)
            model = model.to(device)

        if Model == 'VGG13':
            model = net.VGG13()
            model.classifier[6] = nn.Linear(4096, 100)
            model = model.to(device)

        if Model == 'VGG16':
            model = net.VGG16()
            model.classifier[6] = nn.Linear(4096, 100)
            model = model.to(device)

        if Model == 'VGG19':
            model = net.VGG19()
            model.classifier[6] = nn.Linear(4096, 100)
            model = model.to(device)

        if Model == 'GoogleNet':
            model = net.googlenet()
            #            model.fc = nn.Linear(1024, 100)
            model = model.to(device)

        if Model == 'ResNet50':
            model = net.resnet50()
            model = model.to(device)

        transform_alexnet = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if Model == 'ZF-NET':
            model = net.ZFNet()
            model.classifier[6] = nn.Linear(4096, 100)
            model = model.to(device)

        transform_imagenet = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, Dataset, 'train'),
            transform=transform_alexnet)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size,
                                                  shuffle=True, num_workers=0)
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(Dataset_dir, Dataset, 'test'),
            transform=transform_alexnet)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size,
                                                 shuffle=False, num_workers=0)


    if Dataset == 'Military':

        if Model == 'LeNet':
            output = 10
            model = net.lenet()
            model.fc3 = nn.Linear(84, output)
            model = model.to(device)

        if Model == 'AlexNet':
            model = net.alexnet(num_classes=10)
            model.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=2)
            model.classifier[6] = nn.Linear(4096, 10)
            model = model.to(device)





            # model = net.alexnet()
            # model.classifier[6] = nn.Linear(4096, 10)
            # model = model.to(device)

        if Model == 'VGG16':
            model = net2.VGG16()
            model = model.to(device)

        if Model == 'VGG19':
            model = net.VGG19()
            model = model.to(device)

        if Model == 'GoogleNet':

            model = net2.googlenet(num_classes=10)
            model = model.to(device)

        transform_google = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        if Model == 'ResNet50':
            model = net.resnet50()
            model = model.to(device)

        transform_imagenet2 = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        transform_imagenet = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        trainset = torchvision.datasets.ImageFolder(root=os.path.join(Dataset_dir, Dataset, 'train'),
                                                    transform=transform_imagenet)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(Dataset_dir, Dataset, 'test'),
                                                   transform=transform_imagenet)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print('****** Model and Dataset loaded ******')

    # model = models.resnet50(pretrained=True)

    pthfile = r'/data/Newdisk/zhaoxiaoming/steal/sjs/differential_privacy_docker/steal_model_for_origin/Military/alexnet/alexnet_label_query_Military_steal.pth'
    # for w in torch.load(pthfile):
    #     print(w)
    # exit()
    model.load_state_dict(torch.load(pthfile))
    model.cuda()

    # train the model

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    # last = 0
    # hostlist_set = []
    #
    # for epoch in range(args.epoch):
    #     print('\nEpoch: %d' % (epoch + 1))
    #     model.train()
    #     sum_loss = 0.0
    #     correct = 0.0
    #     total = 0.0
    #     for i, data in enumerate(trainloader, 0):
    #         # prepare dataset
    #         length = len(trainloader)
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #
    #         # forward & backward
    #         outputs = model(inputs)
    #         # outputs, aux2, aux1 = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print ac & loss in each batch
    #         sum_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += predicted.eq(labels.data).cpu().sum()
    #         print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
    #               % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
    #         # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    #         # torch.save(model.state_dict(),'/public/zly/zrj/new/ZeroQ-master/vgg16_mini_imagenet_10_bd.pth')
    #
    #         # Return_FLAG = ws_client.check()
    #         # if Return_FLAG:
    #         #     pass
    #         # else:
    #         #     exit()
    #
    #     # get the ac with valdataset in each epoch
    #     print('Waiting Test...')
    #     with torch.no_grad():
    #         correct = 0
    #         total = 0
    #         num = 0
    #         for i, data in enumerate(testloader, 0):
    #             num = num + 1
    #             model.eval()
    #             images, labels = data
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = model(images)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum()
    #             test_loss = criterion(outputs, labels)
    #             sum_loss += test_loss.item()
    #             # print('Test\'s ac is: %.3f%%' % (100 * torch.true_divide(correct, total)))
    #             final_loss = round(sum_loss / (i + 1), 2)
    #     now = round((100. * correct / total).item(), 2)
    #     jsontext = {
    #         "epoch": epoch + 1,
    #         "Current_Loss": final_loss,
    #         "Current_Accuracy": now
    #     }
    #     hostlist_set.append(jsontext)
    #     if args.json:
    #         jsondata = json.dumps(hostlist_set, ensure_ascii=False, sort_keys=False, indent=2)
    #         f = open(
    #             '/data0/BigPlatform/ZJPlatform/010-ALLData/000_JSON/Image_Train_{}_{}.json'.format(Dataset, Dataset),
    #             'w')
    #         f.write(jsondata)
    #         f.close()
    #     # else:
    #     #     # 发送结果,并判断返回值
    #     #
    #     #     Return_FLAG = ws_client.send(jsontext)
    #     #     if Return_FLAG:
    #     #         pass
    #     #     else:
    #     #         exit()
    #
    #     if now > last:
    #         torch.save(model.state_dict(),
    #                    '/data0/BigPlatform/zxm/results/ImageNet-100/normal_model/{}_{}.pth'.format(Model, Dataset))
    #         print(now)
    #         last = now

    # Test the model
    correct = 0
    total = 0
    start_time = time.time()  # 起始时间
    for data in testloader:
        model.eval()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        # with open('2.txt', 'a+') as f:
        #     for x in labels.detach().cpu().numpy().tolist():
        #         f.write(f'{str(x)}')
        #     f.write('\n')
        outputs = model(images, 'aaa')
        # with open('1.txt', 'a+') as f:
        #     for x in outputs.detach().cpu().numpy().reshape(-1).tolist():
        #         f.write(f'{str(round(x, 2))}')
        #     f.write('\n')
        _, predicted = torch.max(outputs.data, 1)
        # with open('3.txt', 'a+') as f:
        #     for x in predicted.detach().cpu().numpy().tolist():
        #         f.write(f'{str(x)}')
        #     f.write('\n')
        total += labels.size(0)
        correct += (predicted == labels).sum()
        print('Test\'s ac is: %.3f%%' % (100 * torch.true_divide(correct, total)))
    end_time = time.time()  # 结束运行时间
    print("总样本数量：", total)
    print("测试准确率：", correct / total)
    print('cost %f second' % (end_time - start_time))

    print('****** test finished ******')


if __name__ == "__main__":
    # args = arg_parse()
    Train("Military", "AlexNet")