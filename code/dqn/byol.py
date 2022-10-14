from PIL import Image
import torch
from torchvision import models
from byol_pytorch import BYOL
import numpy as np
import torch.nn.functional as F
from sklearn.manifold import TSNE
import pandas as pd
from torchvision import transforms
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from dqn.buffer_dataset import BufferDataset


class BYOL_(object):
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        resnet = models.resnet50(pretrained=True)
        resnet.conv1 = torch.nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.learner = BYOL(
            resnet,
            image_size = 64,
            hidden_layer = 'avgpool'
        ).cuda()

        self.opt = torch.optim.Adam(self.learner.parameters(), lr=3e-4)
        self.features = []
        
        self.toPIL = transforms.ToPILImage()
        self.count = 0
        self.m_featur = None

    def cal(self, obs): # 计算样本和样本池中的特征距离(平均余弦距离)
        self.learner.eval()
        obs = np.expand_dims(obs, axis=0)
        # obs = np.expand_dims(np.mean(obs, axis=1), axis=1) # .repeat(3, axis=1)
        # print(f"obs shape is {obs.shape}")
        obs = torch.tensor(obs, dtype=torch.float32).cuda()
        projection, embedding = self.learner(obs, return_embedding = True)
        embedding = F.normalize(embedding.squeeze(0), p=2, dim=0).detach().cpu()
        # print(embedding)
        # print(f"pool feature is {self.features[0]}")
        # print(f"embedding is {embedding}")
        distance = torch.mm(self.features, embedding.reshape(-1, 1))
        # print(f"distance shape is {distance}")
        distance = torch.mean(distance).numpy() # 已经求均值了
        # avg_dist = 0
        # for fea in self.features:
        #     dist = torch.sqrt(torch.sum(torch.square(fea - self.m_feature)))
        #     avg_dist += dist
        print(f"distance is {1 - distance}")
        return distance
        

    def cal_all(self, replaybuffer, epochs=0):
        self.learner.eval()
        self.features = []
        for idx in range(replaybuffer.__len__()):
            obs = replaybuffer._encode_sample([idx], True)[0]
            # obs = np.expand_dims(np.mean(obs, axis=1), axis=1) # .repeat(3, axis=1)
            # print(f"obs shape is {obs.shape}")
            obs = torch.tensor(obs, dtype=torch.float32).cuda()
            # print(f"obs shape is {obs.shape}")
            with torch.no_grad():
                projection, embedding = self.learner(obs, return_embedding = True)
                embedding = F.normalize(embedding.squeeze(0), p=2, dim=0)
                # print(f"projection shape is {projection.shape}")
                # print(embedding)
                embedding = embedding.cpu()
                self.features.append(embedding.clone())
                del obs
                torch.cuda.empty_cache()
            # del projection
            # del embedding
        self.features = torch.stack(self.features)
        # torch.save(self.features, f"logs/{epochs}-{self.count}.pth")
        pol_average_distance = 0
        dist = []
        for i in range(len(self.features)):
            sample = self.features[i]
            cos_val = torch.mean(torch.mm(self.features, sample.reshape(-1, 1))).numpy()
            pol_average_distance += (1 - cos_val)
        #     dist.append(1-cos_val)
            
        # idxs = np.where(dist <= np.percentile(dist, 99.5))   # 把离群点的特征也剔除
        # self.features = self.features[idxs]
        # for i in range(len(self.features)):
        #     sample = self.features[i]
        #     cos_val = torch.mean(torch.mm(self.features, sample.reshape(-1, 1))).numpy()
        #     pol_average_distance += (1 - cos_val)
        
        pol_average_distance = pol_average_distance / len(self.features)
        # new_dist = [ele for ele in dist if ele < np.percentile(dist, 99)]
        # pol_average_distance = np.mean(new_dist)
        # vis_feas = self.features.numpy()

        # self.m_feature = torch.mean(self.features, dim=0)
        # for i in range(len(self.features)):
        #     sample = self.features[i]
        #     # cos_val = torch.mean(torch.mm(self.features, sample.reshape(-1, 1))).numpy()
        #     dist = torch.sqrt(torch.sum(torch.square(sample - mean_feature)))
        #     # pol_average_distance += (1 - cos_val)
        #     pol_average_distance += dist
        # # vis_feas = self.features.numpy()

        # tsne_obj = TSNE(n_components=3).fit_transform(vis_feas) # 可视化特征池
        # x_min, x_max = np.min(tsne_obj, 0), np.max(tsne_obj, 0)
        # tsne_obj = tsne_obj / (x_max - x_min)
        # fig = plt.figure()
        # ax = Axes3D(fig)
        # ax.scatter(tsne_obj[:, 0], tsne_obj[:, 1], tsne_obj[:, 2], s=0.5)

        # 关闭了plot的坐标显示
        # plt.axis('off')


        # tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
        #                 'Y':tsne_obj[:,1]})
        # sns.scatterplot(x="X", y="Y",
        #       palette=['green','red','yellow','blue'],
        #       legend='full',
        #       data=tsne_df)

        # plt.savefig(f"test_figures/{epochs}-{self.count}-features.jpg")
        print(f"feature shape is {self.features.shape} , pol_average_distance is {pol_average_distance}")
        self.count += 1
        return pol_average_distance



    def train(self, replaybuffer, epochs):
        self.learner.train()
        # images = replaybuffer.sample(self.batch_size, True)[0]
        # # images = np.mean(images, axis=1)
        # # print(images.shape)
        # for i in range(len(images)):
        #     pic = Image.fromarray(np.uint8(images[i][0]))
        #     pic.save(f"test_figures/Qbert_grey_batch_{i}_0.jpg")
        #     pic = Image.fromarray(np.uint8(images[i][1]))
        #     pic.save(f"test_figures/Qbert_grey_batch_{i}_1.jpg")     
        #     pic = Image.fromarray(np.uint8(images[i][2]))
        #     pic.save(f"test_figures/Qbert_grey_batch_{i}_2.jpg") 
        #     pic = Image.fromarray(np.uint8(images[i][3]))
        #     pic.save(f"test_figures/Qbert_grey_batch_{i}_3.jpg") 

        buffer_dataset = BufferDataset(replaybuffer)
        buffer_loader = DataLoader(buffer_dataset, batch_size=self.batch_size, shuffle=True)
        ep = 0
        for _ in range(max(epochs // 2**(self.count), 10)):
            epochs_loss = 0
            for  idx, images in enumerate(buffer_loader):
            # for idx in range(replaybuffer.__len__() // self.batch_size):
                # images = replaybuffer._encode_sample([item for item in range(idx, idx + self.batch_size\
                #      if idx + self.batch_size < replaybuffer.__len__() else replaybuffer.__len__())], True)[0]
                # images = replaybuffer.sample(self.batch_size, True)[0]
                # images = np.mean(images, axis=1)
                # images = np.expand_dims(images, axis=1) #.repeat(3, axis=1)
                # images = torch.tensor(images, dtype=torch.float32)
            # images = replaybuffer.sample(self.batch_size, True)[0]
            # images = np.mean(images, axis=1)
            # # print(images.shape)
            # # for i in range(len(images)):
            # #     pic = Image.fromarray(np.uint8(images[i]))
            # #     pic.save(f"test_figures/grey_batch_{i}.jpg")    
            # images = np.expand_dims(images, axis=1).repeat(3, axis=1)

            # images = torch.tensor(images, dtype=torch.float32).cuda()
                # for i in range(len(images)):
                #     pic = self.toPIL(images[i])
                #     pic.save(f"test_figures/Qbert_batch_{i}.jpg")
                # print(images.shape)
                images = images.cuda()
            # print(f"images shape is {images.shape}")
                loss = self.learner(images)
                # print(f"contrstive loss is : {loss.item()}")
                epochs_loss += loss.item()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                self.learner.update_moving_average() # update moving average of target encoder

                # del images
                ep += 1
            print(f"epoch average loss is {epochs_loss / ep}")
        print("call cal alll")
        pol_average_distance = self.cal_all(replaybuffer, epochs)
        return pol_average_distance

