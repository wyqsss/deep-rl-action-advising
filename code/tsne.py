import torch
from sklearn.manifold import TSNE
import pandas as pd
from torchvision import transforms
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import numpy as np
import random

vis_feas = torch.load("logs/10-0.pth")
vis_feas = vis_feas.numpy()

tsne_obj = TSNE(n_components=3).fit_transform(vis_feas) # 可视化特征池
x_min, x_max = np.min(tsne_obj, 0), np.max(tsne_obj, 0)
tsne_obj = tsne_obj / (x_max - x_min)

advised = []
for i in range(len(tsne_obj)):
    if random.random() > 0.9:
        advised.append(1)
    else:
        advised.append(0)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(tsne_obj[:, 0], tsne_obj[:, 1], tsne_obj[:, 2], s=0.5, color=advised)

# 关闭了plot的坐标显示
# plt.axis('off')


tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                'Y':tsne_obj[:,1], 'advised':advised})
sns.scatterplot(x="X", y="Y", s=10,
        # palette=['green','red','yellow','blue'],
        # legend='full',
        style='advised',
        hue='advised',
        data=tsne_df)

plt.savefig(f"test_figures/features.jpg")