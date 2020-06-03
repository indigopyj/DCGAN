import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *
from dataset import *
from utils import *

from torchvision import transforms, datasets


## Parser
parser = argparse.ArgumentParser(description='Train the UNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default='./datasets', type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default='./checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='./log', type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")


parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

parser.add_argument("--task", default="DCGAN", choices=["DCGAN"], type=str, dest="task")
parser.add_argument("--opts", nargs="+", default = ["bilinear", 4, 0], dest="opts")

parser.add_argument('--ny', type=int, default=64, dest='ny')
parser.add_argument('--nx', type=int, default=64, dest='nx')
parser.add_argument('--nch', type=int, default=3, dest='nch')
parser.add_argument('--nker', type=int, default=128, dest='nker')

parser.add_argument("--network", default = "DCGAN", choices=["unet", "resnet", "autoencoder", "srresnet","DCGAN"], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                   

## Parameter
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

#data_dir = '/content/gdrive/My Drive/Colab Notebooks/unet_data/train'
#ckpt_dir = '/content/gdrive/My Drive/Colab Notebooks/unet_data/checkpoints'
#log_dir = '/content/gdrive/My Drive/Colab Notebooks/unet_data/log'

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir

result_dir = args.result_dir
mode = args.mode
train_continue = args.train_continue

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

network = args.network
learning_type = args.learning_type

cmap = None

result_dir_train = os.path.join(result_dir, 'train')
result_dir_test = os.path.join(result_dir, 'test')



if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir_train, 'png'))
    # os.makedirs(os.path.join(result_dir_train, 'numpy'))

    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))


# 네트워크 생성하기
if network == "DCGAN":
    netG = DCGAN(in_channels=100, out_channels=nch, nker=nker).to(device)
    netD = Discriminator(in_channels=nch, out_channels=1, nker=nker).to(device)

    # initialization
    init_weights(netG, init_type="normal", init_gain=0.02)
    init_weights(netD, init_type="normal", init_gain=0.02)




# 로스함수 정의
#fn_loss = nn.BCEWithLogitsLoss().to(device)
#fn_loss = nn.MSELoss().to(device)
fn_loss = nn.BCELoss().to(device)

# optimizer 정의
optimG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))


# Tensorboard를 사용하기 위한 summary writer 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))

# 부수적인 함수 설정
fn_tonumpy = lambda x:x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # from tensor to numpy
fn_denorm = lambda x, mean, std: x*std + mean # denormalization
#fn_class = lambda x: 1.0*(x > 0.5)  # binary classification



if mode=='train':
    # 네트워크 학습하기

    # normalization : range (-1, 1)로 tanh와 동일한 범위로 만들어줌
    transform_train = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])



    #data_dir = '/content/gdrive/My Drive/Colab Notebooks/unet_data/'
    dataset_train = Dataset(data_dir=data_dir, transform=transform_train, task=task, opts=opts)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)


    # 부수적인 변수들 정의
    num_data_train = len(dataset_train)
    num_batch_train = np.ceil(num_data_train / batch_size)

else:
    transform_test = transforms.Compose([Resize(shape=(ny, nx, nch)), Normalization(mean=0.5, std=0.5)])

    #data_dir = '/content/gdrive/My Drive/Colab Notebooks/unet_data/'
    dataset_test = Dataset(data_dir=data_dir, transform=transform_test, task=task, opts=opts)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    # 부수적인 변수들 정의
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)


if mode=='train':
    ############# Training ################
    if train_continue=="on":
        netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)
    else:
        st_epoch = 0 # start epoch number
        
     

    for epoch in range(st_epoch+1, num_epoch + 1):
      netG.train()
      netD.train()

      loss_G_train = []
      loss_D_real_train = []
      loss_D_fake_train = []

      for batch, data in enumerate(loader_train, 1):
        # forward pass

        label = data['label'].to(device) # original image
        input = torch.randn(label.shape[0], 100, 1, 1).to(device)   # noise input(B, C, H, W)


        output = netG(input)

        # backward pass netD
        set_requires_grad(netD, True)
        optimD.zero_grad()

        pred_real = netD(label)
        pred_fake = netD(output.detach())   # Discriminator로부터 오는 backpropagation이 generator로 넘어가지 않게 detach

        loss_D_real = fn_loss(pred_real, torch.ones_like(pred_real))
        loss_D_fake = fn_loss(pred_fake, torch.zeros_like(pred_fake))
        loss_D = 0.5 * (loss_D_real + loss_D_fake) # average

        loss_D.backward()
        optimD.step()

        # backward pass netG
        set_requires_grad(netD, requires_grad=False)
        optimG.zero_grad()

        pred_fake = netD(output)    # Discriminator로부터 오는 backpropagation이 generator로 넘어가야 학습됨

        loss_G = fn_loss(pred_fake, torch.ones_like(pred_fake))

        loss_G.backward()
        optimG.step()

        # loss function 계산
        loss_G_train += [loss_G.item()]
        loss_D_real_train += [loss_D_real.item()]
        loss_D_fake_train += [loss_D_fake.item()]

        print("TRAIN : EPOCH %04d / %04d | BATCH %04d / %04d | "
            "GEN : %.4f | DISC REAL : %.4f | DISC FAKE : %.4f" %
              (epoch, num_epoch, batch, num_batch_train, np.mean(loss_G_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))

        if batch % 20 == 0:

            # tanh로 (-1,1) 범위를 가졌던 것을 (0,1)로 바꿔줌
            # png로 저장하기 위해서 이미지의 range를 0~1로 클리핑
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()
            output = np.clip(output, a_min=0, a_max=1)

            save_id =  num_batch_train * (epoch - 1) + batch
            plt.imsave(os.path.join(result_dir_train, "png", "%04d_output.png" %save_id), output[0].squeeze(), cmap=cmap)

            writer_train.add_image('output', output, save_id, dataformats='NHWC')

    writer_train.add_scalar('loss_G', np.mean(loss_G_train), epoch)
    writer_train.add_scalar('loss_D_real', np.mean(loss_D_real_train), epoch)
    writer_train.add_scalar('loss_D_fake', np.mean(loss_D_fake_train), epoch)

    if epoch % 2 == 0 or epoch == num_epoch:
        save(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD, epoch=epoch)




    writer_train.close()

else:
    ############# Test ################

    st_epoch = 0  # start epoch number
    netG, netD, optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir, netG=netG, netD=netD, optimG=optimG, optimD=optimD)


    with torch.no_grad():
      # Generator 만 test함
      netG.eval()

      input = torch.randn(batch_size, 100, 1, 1).to(device)
      output = netG(input)

      output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

      for j in range(output.shape[0]): # 결과 저장

        save_id = j

        output_ = output[j]
        # png로 저장하기 위해서 이미지의 range를 0~1로 클리핑
        output_ = np.clip(output_, a_min=0, a_max=1)
        plt.imsave(os.path.join(result_dir_test, 'png', "%04d_output.png" % save_id), output_)
          
          




