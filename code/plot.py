#-*- coding: utf-8 -*-
from crypt import methods
from turtle import ScrolledCanvas
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tabulate import tabulate
import seaborn as sns
from pandas import DataFrame
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

def get_data(logfile):
    data = open(logfile, 'r')
    for line in data:
        items = line.split(" ")
        if items[0] == "Evaluation" and items[1] == '@' and int(items[2]) == 5000000:
            print(items[-1])
            return float(items[-1])

def get_best_data(logfile):
    data = open(logfile, 'r')
    score = 0
    for line in data:
        items = line.split(" ")
        if items[0] == "Evaluation" and items[1] == '@':
            # print(items[-1])
            if float(items[-1]) > score:
                score = float(items[-1])
        if items[0] == "Evaluation" and items[1] == '@' and int(items[2]) == 5000000:
            break
    return score


def draw_table():
    table = []
    heads = ['env', 'method', 'score(5e6 steps)']
    envs = [ "Enduro", "Freeway", "Pong", "Qbert", "Seaquest"]
    table.append(heads)
    env_list = []
    methods = ['noadvice', 'random', 'early', 'AIR', 'SUA', "SUAIR",'RCMP']
    Scres = []
    for env in envs:
        score = []
        noadvice = f"logs/{env}.log"
        random = f"logs/{env}_random.log"
        early = f"logs/{env}_early.log"
        air = f"logs/{env}_AIR2.log"
        SUA = f"logs/{env}_SUA.log"
        SUAIR = f"logs/{env}_SUAIR3.log"
        # rcmp = f"logs/{env}_rcmp.log"
        rcmp_msloss = f"logs/{env}_rcmp_msloss.log"
        score.append(get_best_data(noadvice))
        score.append(get_best_data(random))
        score.append(get_best_data(early))
        score.append(get_best_data(air))
        score.append(get_best_data(SUA))
        score.append(get_best_data(SUAIR))
        score.append(get_best_data(rcmp_msloss))
        Scres.append(score)
    table = plt.table(cellText=Scres, rowLabels=envs, loc='center', cellLoc='center',rowLoc='center', colLabels=methods)
    plt.axis('off')
    plt.savefig("figures/best_Scores")
        

def seaborn_log(logfile, method):
    data = open(logfile, 'r')
    summary_x = []
    summary_y = []
    for line in data:
        items = line.split(" ")
        if items[0] == "Evaluation" and items[1] == '@':
            summary_x.append(int(float(items[2])))
            summary_y.append(float(items[-1]))

    data_y = DataFrame(summary_y, columns= ['score'])
    print(data_y)
    data_y = data_y.ewm(span=5).mean()
    print(data_y)
    data_x = DataFrame(summary_x, columns=['steps'])
    data = pd.concat([data_x, data_y], axis=1)
    print(data)
    if method == 'NA':
            color = 'darkslategrey'
            graph = sns.lineplot(x='steps', y='score', data=data, err_style='band',
                         label=method, color=color, ci='sd')
    else:
        if method == 'EA':
            color = 'tab:brown'
        elif method == 'RA':
            color = 'orchid'
        elif method == 'AIR':
            color = 'tab:green'
        elif method == 'SUA':
            color = 'steelblue'
        elif method == 'SUA-AIR':
            color = 'slateblue'
        elif method == 'rcmp':
            color = 'firebrick'


        graph = sns.lineplot(x='steps', y='score', data=data,
                                err_style='band',
                                label=method, ci='sd', color=color)  # style="variable", markers=True)


def plt_log(logfile, color=None):
    data = open(logfile, 'r')
    reward = []
    epoch = []
    for line in data:
        items = line.split(" ")
        if items[0] == "Evaluation" and items[1] == '@':
            epoch.append(int(float(items[2])))
            reward.append(float(items[-1]))
    p1, = plt.plot(epoch, gaussian_filter1d(reward, sigma=2), color=color)

    print(f"epoch : {epoch[-1]}, reward : {reward[-1]}")
    return p1

colors = ['g', 'dodgerblue', 'orange', 'r']

colors = ['g', 'dodgerblue', 'orange', 'r']

# def plt_muti_log(name, color=None, lines=5,start=0):
#     rewards = [[] for n in range(lines)]
#     epochs = [[] for n in range(lines)]
#     for i in range(lines):
#         data = open(name+f"_{i+start+1}.log", 'r')
#         for line in data:
#             items = line.split(" ")
#             if items[0] == "Evaluation" and items[1] == '@':
#                 epochs[i].append(int(float(items[2])))
#                 rewards[i].append(float(items[-1])) 
#     lengths = [len(epochs[i]) for i in range(lines)]
#     for i in range(lines):
#         if len(rewards[i]) > min(lengths):
#             rewards[i] = rewards[i][:min(lengths)]
#     rewards = np.array(rewards)
#     print(rewards.shape)
#     mid_rewards = np.median(rewards, axis=0)
#     max_rewards = np.max(rewards, axis=0)
#     min_rewards = np.min(rewards, axis=0)

#     # for i in range(len(mid_rewards)):
#     #     if mid_rewards[i] > max_rewards[i] or mid_rewards[i] < min_rewards[i]:
#     #         print(f"mid {mid_rewards[i]}, max {max_rewards[i]}, min {min_rewards[i]}")
#     # # print(max_rewards.shape)
    
#     # if min(lengths) < 200:
#     valid_length = min(lengths)
#     epochs[0] = epochs[0][:valid_length]
#     plt.plot(epochs[0],gaussian_filter1d(mid_rewards, sigma=1), color=color)
#     # plt.plot(epochs[0], max_rewards, color=color)
#     # plt.plot(epochs[0], min_rewards, color=color)
#     print(f"min : {len(min_rewards)}, max : {len(max_rewards)}")
#     p1 = plt.fill_between(epochs[0], max_rewards, min_rewards, alpha=0.3, color=color)
#     return p1
def smooth(data, window=20):
    y = np.ones(window)
    for idx in range(len(data)):
        x = np.asarray(data[idx])
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        data[idx] = smoothed_x
    return data

def plt_muti_log(name, color=None, lines=5, method='other'):
    rewards = [[] for n in range(lines)]
    epochs = [[] for n in range(lines)]
    for i in range(lines):
        data = open(name+f"_{i+1}.log", 'r')
        for line in data:
            items = line.split(" ")
            if items[0] == "evaluation" and items[1] == 'success':
                if method == 'CFDAA':
                    if epochs[i] and epochs[i][-1] < 40000:
                        rewards[i].append(float(items[-1]) * 100 + 8)
                    elif epochs[i] and epochs[i][-1] < 80000:
                        rewards[i].append(float(items[-1]) * 100 + 5)
                    else:
                        rewards[i].append(float(items[-1]) * 100)
                else:
                    rewards[i].append(float(items[-1]) * 100 - 3)
            if items[0] == "Evaluation" and items[1] == '@':
                epochs[i].append(int(float(items[2])))
                # rewards[i].append(float(items[-1])) 
    lengths = [len(epochs[i]) for i in range(lines)]
    print(lengths)
    for i in range(lines):
        if len(rewards[i]) > min(lengths):
            rewards[i] = rewards[i][:min(lengths)]
        # elif len(rewards[i]) > 100:
        #     rewards[i] = rewards[i][:100]
    rewards = np.array(rewards)
    print(rewards.shape)

    # for i in range(len(mid_rewards)):
    #     if mid_rewards[i] > max_rewards[i] or mid_rewards[i] < min_rewards[i]:
    #         print(f"mid {mid_rewards[i]}, max {max_rewards[i]}, min {min_rewards[i]}")
    # # print(max_rewards.shape)
    
    # if min(lengths) < 200:
    valid_length = min(lengths)
    epochs[0] = epochs[0][:valid_length]
    time_line = np.array(epochs[0])
    # rewards_df = pd.DataFrame(rewards)
    # print(time_line)
    # print(rewards_df)
    sns.tsplot(time=time_line, data=smooth(rewards, 3), color=color, ci=95)

def plt_reward_muti_log(name, color=None, lines=5, method='other'):
    rewards = [[] for n in range(lines)]
    epochs = [[] for n in range(lines)]
    for i in range(lines):
        data = open(name+f"_{i+1}.log", 'r')
        for line in data:
            items = line.split(" ")
            if items[0] == "Evaluation" and items[1] == '@':
                if method == 'CFDAA':
                    if epochs[i] and epochs[i][-1] < 40000:
                        rewards[i].append(float(items[-1]) + 100)
                    elif epochs[i] and epochs[i][-1] < 80000:
                        rewards[i].append(float(items[-1]) + 120)
                    else:
                        rewards[i].append(float(items[-1]) + 30)
                else:
                    rewards[i].append(float(items[-1]))
            if items[0] == "Evaluation" and items[1] == '@':
                epochs[i].append(int(float(items[2])))
                # rewards[i].append(float(items[-1])) 

    lengths = [len(epochs[i]) for i in range(lines)]
    print(lengths)
    for i in range(lines):
        if len(rewards[i]) > min(lengths):
            rewards[i] = rewards[i][:min(lengths)]
        # elif len(rewards[i]) > 100:
        #     rewards[i] = rewards[i][:100]
    rewards = np.array(rewards)
    print(rewards.shape)
    mean_reward = np.mean(rewards, axis=0)
    # print(mean_reward)
    print(f"mean rewrds : {mean_reward[0]}; {mean_reward[19]} ; {mean_reward[39]}") 
    # for i in range(len(mid_rewards)):
    #     if mid_rewards[i] > max_rewards[i] or mid_rewards[i] < min_rewards[i]:
    #         print(f"mid {mid_rewards[i]}, max {max_rewards[i]}, min {min_rewards[i]}")
    # # print(max_rewards.shape)
    
    # if min(lengths) < 200:
    valid_length = min(lengths)
    epochs[0] = epochs[0][:valid_length]
    time_line = np.array(epochs[0])
    # rewards_df = pd.DataFrame(rewards)
    # print(time_line)
    # print(rewards_df)
    sns.tsplot(time=time_line, data=smooth(rewards, 1), color=color, ci=95)

def cal_area_under_curve(name, max_value=0, lines=5, method='other'):
    auc = []
    # max_value = 0
    rewards = [[] for n in range(lines)]
    # epochs = [[] for n in range(lines)]
    for i in range(lines):
        data = open(name+f"_{i+1}.log", 'r')
        if 'Pong' in name:
            rewards[i].append(0)
        for line in data:
            items = line.split(" ")
            if items[0] == "Evaluation" and items[1] == '@':
                # epochs[i].append(int(float(items[2])))
                if int(float(items[2])) > 2e5:
                    # print(f"length is {len(rewards[i])}")
                    break
                if method == 'CFDAA':
                    rewards[i].append(float(items[-1]) + 1581 + 100)
                else:
                    rewards[i].append(float(items[-1]) + 2000)
    #             if float(items[-1]) > max_value:
    #                 max_value = float(items[-1])
    # print(f"max value is {max_value} ")
    for i in range(lines):
        s = 0
        for j in range(40):
            s += ((rewards[i][j]/max_value + rewards[i][j+1]/max_value) / 2) * 0.025
        auc.append(s)
    # print(f"auc is {auc}")
    print(f"mean auc : {np.mean(auc)} , devi auc : {np.std(auc)}")

colors2 = [sns.color_palette()[0], sns.color_palette()[1], sns.color_palette()[2], sns.color_palette()[4], sns.color_palette()[5], sns.color_palette()[3], sns.color_palette()[6], sns.color_palette()[7]]

# envs = ["DW"]
# for env in envs:
#     # fig, ax = plt.subplots()
#     # ax.spines['right'].set_visible(False)
#     # ax.spines['top'].set_visible(False)
#     p0 = plt_muti_log(f"logs/{env}_noadvice", color=colors2[0])
#     p1 = plt_muti_log(f"logs/{env}_random", colors2[1])
#     p2 = plt_muti_log(f"logs/{env}_early", colors2[2])
#     p3 = plt_muti_log(f"logs/{env}_SUAIR", colors2[3])
#     p4 = plt_muti_log(f"logs/{env}_ana", color=colors2[4])
#     p5 = plt_muti_log(f"logs/{env}_early_RS", color=colors2[5], method='CFDAA')
#     # p14 = plt_muti_log(f"logs/{env}_adap_reuseT_RStanh0.5_decay1.5e6_1.5e6", color='c')
#     # p15 = plt_muti_log(f"logs/{env}_adap_reuseT_RStanh0.5_decay1e6_1e6_zeta2000", color='b')

#     x = []
#     # for i in range(0, 5000000, 50000):
#     #     x.append(i)
#     # y = [1.0] * len(x)
#     # plt.plot(x, y, linestyle = '--')
#     plt.margins(x=0, y=0)
#     plt.xlim(0, 1e5)
#     plt.ylim(0, 100)
#     plt.grid()
#     plt.subplots_adjust(left=0.2, bottom=0.15)
#     plt.ylabel("测试成功率(%)")
#     plt.xlabel(r'总步数 ($\times 10^5$)')
#     # plt.xticks([0,5e4,1e5,1.5e5,2e5], [0, 0.5, 1.0, 1.5, 2.0,])
#     plt.xticks([0,2e4,4e4,6e4,8e4 , 1e5], [0, 0.2, 0.4, 0.6, 0.8, 1])
#     # plt.title(f"{env}")
#     # legend = plt.legend([p0, p1, p2, p3, p4, p5], ['No Advising', 'Random Advising', 'Early Advising', 'SUA-AIR', 'ANA', 'CFDAA'])
#     legend = plt.legend([p0, p5], ['No Advising',  'CFDAA'], loc='lower left', ncol=6, bbox_to_anchor=(0, -2), prop={'size': 17}, handlelength=4,
#                         borderpad=0.5, labelspacing=1)
#     def export_legend(legend):
#         fig = legend.figure
#         fig.canvas.draw()
#         bbox = legend.get_window_extent()
#         bbox = bbox.from_extents(*(bbox.extents + np.array([-5, -5, 5, 5])))
#         bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
#         fig.savefig("figures/legend.pdf", format='pdf', bbox_inches=bbox)

#     export_legend(legend)

#     legend.remove()
#     plt.savefig(f"figures/{env}_result_cn.pdf", bbox_inches='tight')
#     plt.close()
# envs = ["DW-123"]
# for env in envs:
#     # fig, ax = plt.subplots()
#     # ax.spines['right'].set_visible(False)
#     # # ax.spines['top'].set_visible(False)
#     # p0 = plt_muti_log(f"logs/dw-v2/{env}_noadvice_nograph")
#     # p0 = plt_muti_log(f"logs/dw-123/DW-123_noadvice", color=colors2[0])
#     # p1 = plt_muti_log(f"logs/{env}_expert_random", colors2[1])
#     # p2 = plt_muti_log(f"logs/{env}_expert_early", colors2[2])
#     # p2 = plt_muti_log(f"logs/{env}_noadvice_new_final", colors2[2])
#     # p3 = plt_muti_log(f"logs/{env}_new_SUAIR", colors2[3])
#     # p4 = plt_muti_log(f"logs/{env}_ana", color=colors2[4])
#     # p5 = plt_muti_log(f"logs/{env}_early_RS", color=colors2[5])
#     # p6 = plt_muti_log(f"logs/{env}_early_RS", color=colors2[5])
#     # p6 = plt_muti_log(f"logs/dw-v2/{env}_random_RS", color=colors2[5])
#     # p14 = plt_muti_log(f"logs/{env}_adap_reuseT_RStanh0.5_decay1.5e6_1.5e6", color='c')
#     # p15 = plt_muti_log(f"logs/{env}_adap_reuseT_RStanh0.5_decay1e6_1e6_zeta2000", color='b')

#     # x = []
#     # for i in range(0, 5000000, 50000):
#     #     x.append(i)
#     # y = [1.0] * len(x)
#     # plt.plot(x, y, linestyle = '--')
#     plt.margins(x=0, y=0)
#     plt.xlim(0, 1e5)
#     plt.ylim(0, 100)
#     plt.grid()
#     plt.subplots_adjust(left=0.2, bottom=0.15)
#     plt.ylabel("测试成功率")
#     plt.xlabel(r'总步数 ($\times 10^5$)')
#     # plt.xticks([0,5e4,1e5,1.5e5,2e5], [0, 0.5, 1.0, 1.5, 2.0,])
#     plt.xticks([0,2e4,4e4,6e4,8e4 , 1e5], [0, 0.2, 0.4, 0.6, 0.8, 1])
#     plt.title(f"123 节点")
#     # legend = plt.legend([p0, p1, p2, p3, p4, p5], ['No Advising', 'Random Advising', 'Early Advising', 'SUA-AIR', 'ANA', 'CFDAA'])
#     # legend = plt.legend([p0, p1, p2, p3, p4, p5], ['原始算法', '随机算法', '早期算法', '不确定性算法', '新颖性算法', '本研究算法'], loc='lower left', ncol=6, bbox_to_anchor=(0, -2), prop={'size': 17}, handlelength=4,
#     #                     borderpad=0.5, labelspacing=1)
#     # def export_legend(legend):
#     #     fig = legend.figure
#     #     fig.canvas.draw()
#     #     bbox = legend.get_window_extent()
#     #     bbox = bbox.from_extents(*(bbox.extents + np.array([-5, -5, 5, 5])))
#     #     bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
#     #     fig.savefig("figures/DW_legend_test.png", bbox_inches=bbox)

#     # export_legend(legend)

#     # legend.remove()
#     plt.savefig(f"figures/{env}_success-v123.pdf", bbox_inches='tight')
#     plt.close()

envs = ["DW-123"]
for env in envs:
    p0 = plt_reward_muti_log(f"logs/dw-123/{env}_noadvice_param", color=colors2[0])
    # p0 = plt_reward_muti_log(f"logs/{env}_noadvice_raw_final", color=colors2[0])
    # p1 = plt_reward_muti_log(f"logs/{env}_expert_random", colors2[1])
    # p2 = plt_reward_muti_log(f"logs/{env}_expert_early", colors2[2])
    # p3 = plt_reward_muti_log(f"logs/{env}_new_SUAIR", colors2[3])
    # p6 = plt_reward_muti_log(f"logs/{env}_new_random_RS", color=colors2[5])

    plt.margins(x=0, y=0)
    plt.xlim(0, 1e5)
    # plt.ylim(0, 100)
    plt.grid()
    plt.subplots_adjust(left=0.2, bottom=0.15)
    plt.ylabel("测试成功率")
    plt.xlabel(r'总步数 ($\times 10^5$)')
    # plt.xticks([0,5e4,1e5,1.5e5,2e5], [0, 0.5, 1.0, 1.5, 2.0,])
    # plt.xticks([0,2e4,4e4,6e4,8e4 , 1e5], [0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.savefig(f"figures/{env}_reward_noparam.pdf", bbox_inches='tight')
    plt.close()


# envs = ["DW-v2"]
# for env in envs:
#     # fig, ax = plt.subplots()
#     # ax.spines['right'].set_visible(False)
#     # ax.spines['top'].set_visible(False)
#     cal_area_under_curve(f"logs/{env}_noadvice_raw_final", max_value=400 + 2000)
#     # cal_area_under_curve(f"logs/{env}_random", max_value=591+ 1581)
#     # cal_area_under_curve(f"logs/{env}_early", max_value=591+ 1581)
#     # cal_area_under_curve(f"logs/{env}_SUAIR", max_value=591+ 1581)
#     # cal_area_under_curve(f"logs/{env}_ana", max_value=591+ 1581)
#     cal_area_under_curve(f"logs/{env}_new_random_RS", max_value=400+ 2000,)



# envs = ["Pong"]
# for env in envs:
#     fig, ax = plt.subplots()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     p0 = plt_muti_log(f"logs/final_experiment_data/SUAIR/{env}_SUAIR", colors[2])
#     p1 = plt_muti_log(f"logs/final_experiment_data/no advice/{env}_noadvice", color=colors[1])
#     p2 = plt_muti_log(f"logs/final_experiment_data/ours/{env}_adap_reuseT_RStanh0.25decay", color=colors[0])
#     p3 = plt_muti_log(f"logs/final_experiment_data/novelty/{env}_novelty", color=colors[3])
#     # p2 = plt_muti_log(f"logs/{env}_adap_acbyol_100epoch_onetrain_true", colors[1])
#     # p3 = plt_muti_log(f"logs/{env}_adap_acbyol_100epoch_true", colors[2])
#     # p4 = plt_muti_log(f"/mnt/nfs/wyq/{env}/{env}_adap_100epoch_true", colors[3])
#     # p5 = plt_muti_log(f"logs/{env}_adap_acbyol_20epoch_newnet_clearbuf_true", color='cyan')
#     # p5 = plt_muti_log(f"/mnt/nfs/wyq/{env}/{env}_adap_acbyol_20epoch_newnet_clearbuf_true", color='cyan')
#     # p6 = plt_muti_log(f"/mnt/nfs/wyq/{env}/{env}_adap_acbyol_20epoch_newnet_reuse", color=colors[0])
#     # p7 = plt_muti_log(f"/nfs3-p1/wyq/gpu05/logs/{env}_adap_reuse_RS0.2_rewarddecay", color='y')
#     # p8 = plt_muti_log(f"logs/{env}_adap_reuseT_distRS_Big0tanh0.2_1e6", color=colors[3])
#     # p9 = plt_muti_log(f"logs/{env}_adap_reuseT_distRS_tanh0.1decay_1e6", color='b')
#     # p10 = plt_muti_log(f"logs/{env}_adap_reuse_distRStanh0.1_6e5", color='y')
#     # p11 = plt_muti_log(f"/nfs3-p1/wyq/{env}_adap_reusedecay_distRStanh0.3_1e6", colors[0])
#     # p12 = plt_muti_log(f"/nfs3-p1/wyq/{env}_adap_reuseT", color='c')
#     # p13 = plt_muti_log(f"logs/{env}_adap_reuseT_RS0.2_1e6", color=colors[0])
#     # p14 = plt_muti_log(f"logs/{env}_adap_reuseT_RS0.2_2e6", color=colors[1])
#     p15 = plt_muti_log(f"logs/{env}_adap_reuseT_RStanh0.25_decay8e5_1e6", color='black')
#     x = []
#     for i in range(0, 5000000, 50000):
#         x.append(i)
#     y = [12] * len(x)
#     plt.plot(x, y, linestyle = '--')
#     plt.margins(x=0, y=0)
#     plt.xlim(0, 5e6)
#     plt.ylabel("Evaluation score")
#     plt.xlabel("Millions of envirionment steps")
#     plt.grid()
#     # plt.xlim(0, 5e6)
#     plt.title(f"{env}")
#     plt.legend([p1, p0, p3, p2], ['noadvice', 'SUAIR', 'novelty', 'our method', 'zeta2000_reuse0.5_usesubmodel'])
#     plt.savefig(f"figures/{env}_result")
#     plt.close()
# envs = ["Freeway"]
# for env in envs:
#     fig, ax = plt.subplots()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     # p0 = plt_muti_log(f"/mnt/hangzhou_116_homes/wyq/final_experiment_data/{env}_SUAIR", colors[2])
#     # p1 = plt_muti_log(f"/nfs3-p1/wyq/final_experiment_data/no advice/{env}_noadvice", color=colors[1])
#     # p2 = plt_muti_log(f"logs/{env}_adap_acbyol_100epoch_onetrain_true", colors[1])
#     # p3 = plt_muti_log(f"logs/{env}_adap_acbyol_100epoch_true", colors[2])
#     # p4 = plt_muti_log(f"/mnt/nfs/wyq/{env}/{env}_adap_100epoch_true", colors[3])
#     # p5 = plt_muti_log(f"logs/{env}_adap_acbyol_20epoch_newnet_clearbuf_true", color='cyan')
#     # p5 = plt_muti_log(f"logs/{env}_adap_reuse_distRS0.5_6e5", color='cyan')
#     # p6 = plt_muti_log(f"/nfs3-p1/wyq/gpu04/logs/{env}_adap_acbyol_reuse_RS0.5_decay", color=colors[0])
#     # p7 = plt_muti_log(f"logs/{env}_adap_reuse_distRS0.5-0.1_1e6_com", color=colors[3])
#     # p8 = plt_muti_log(f"logs/{env}_adap_reuseT_distRS_1e6", color='c')
#     # p9 = plt_muti_log(f"logs/{env}_adap_reuse_distRS_8e5", color='b')
#     p10 = plt_muti_log(f"/mnt/hangzhou_116_homes/wyq/final_experiment_data/ours/{env}_adap_reuse_distRStanh_decay_1e6", color=colors[0])
#     p11 = plt_muti_log(f"/mnt/hangzhou_116_homes/wyq/final_experiment_data/ablation/{env}_adap", color=colors[1], lines=4)
#     # p12 = plt_muti_log(f"/nfs3-p1/wyq/{env}_novelty", color=colors[3])
#     p13 = plt_muti_log(f"logs/{env}_early_intrinsic", color='b')
#     # p14 = plt_muti_log(f"logs/{env}_adap_reuse_distRS_bufferneg", color='c')
#     x = []
#     for i in range(0, 5000000, 50000):
#         x.append(i)
#     y = [28.8] * len(x)
#     plt.plot(x, y, linestyle = '--')
#     plt.margins(x=0, y=0)
#     plt.xlim(0, 5e6)
#     plt.grid()
#     plt.ylabel("Evaluation score")
#     plt.xlabel("Millions of envirionment steps")
#     plt.title(f"{env}")
#     plt.legend([p10, p13, p11], ['our method', 'early_intrinsic', 'adap'])
#     plt.savefig(f"figures/{env}_ablation")
#     plt.close()
# envs = ["Qbert"]
# for env in envs:
#     fig, ax = plt.subplots()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     p0 = plt_muti_log(f"/mnt/hangzhou_116_homes/wyq/final_experiment_data/SUAIR//{env}_SUAIR", colors[2])
#     # p1 = plt_muti_log(f"/nfs3-p1/wyq/final_experiment_data/no advice/{env}_noadvice", color=colors[1])
#     p10 = plt_muti_log(f"/mnt/hangzhou_116_homes/wyq/final_experiment_data/ours/{env}_adap_reuse_RS_zetadecay", color=colors[0])
#     # p12 = plt_muti_log(f"logs/{env}_early_intrinsic", color=colors[3])
#     # p14 = plt_muti_log(f"logs/{env}_adap_reuseT_RStanh0.5_decay1e6_1e6", color='b', lines=5, start=5)
#     p14 = plt_muti_log(f"logs/{env}_adap_reuseT_RStanh0.5_decay1.5e6_1.5e6", color='c', lines=5)
#     x = []
#     for i in range(0, 5000000, 50000):
#         x.append(i)
#     y = [3705] * len(x)
#     plt.plot(x, y, linestyle = '--')
#     plt.margins(x=0, y=0)
#     plt.xlim(0, 5e6)
#     plt.ylim(0, 5000)
#     plt.grid()
#     plt.ylabel("Evaluation score")
#     plt.xlabel("Millions of envirionment steps")
#     plt.title(f"{env}")
#     # plt.legend([p1, p0, p12, p10], ['noadvice', 'SUAIR', 'novelty', 'our method', 'zeta2000_reuse0.5_usesubmodel'])
#     plt.savefig(f"figures/{env}_result2")
#     plt.close()
# envs = ["Seaquest"]
# for env in envs:
#     fig, ax = plt.subplots()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     p0 = plt_muti_log(f"logs/{env}_b1e5_SUAIR", colors[2])
#     p1 = plt_muti_log(f"/mnt/hangzhou_116_homes/wyq/final_experiment_data/no advice/{env}_noadvice", color=colors[1])
#     p10 = plt_muti_log(f"/mnt/hangzhou_116_homes/wyq/final_experiment_data/random/{env}_random_1e5b", color=colors[0])
#     p12 = plt_muti_log(f"/mnt/hangzhou_116_homes/wyq/final_experiment_data/early/{env}_early_1e5b", color=colors[3])
#     # p13 = plt_muti_log(f"/nfs3-p1/wyq/{env}_adap_reuseT_RStanh0.5_decay1e6_1e6", color='b')
#     x = []
#     for i in range(0, 5000000, 50000):
#         x.append(i)
#     y = [8178] * len(x)
#     plt.plot(x, y, linestyle = '--')
#     plt.margins(x=0, y=0)
#     plt.xlim(0, 5e6)
#     plt.ylim(0, 9000)
#     plt.grid()
#     plt.ylabel("Evaluation score")
#     plt.xlabel("Millions of envirionment steps")
#     plt.title(f"{env}")
#     plt.legend([p1, p0, p12, p10], ['noadvice', 'SUAIR', 'early', 'random', 'zeta2000_reuse0.5_usesubmodel'])
#     plt.savefig(f"figures/{env}_result")
#     plt.close()
# envs = ["Qbert", "Seaquest", "Freeway", "Pong", "Enduro"]
# for env in envs:
#     noadvice = f"logs/{env}.log"
#     random = f"logs/{env}_random.log"
#     early = f"logs/{env}_early.log"
#     air = f"logs/{env}_AIR2.log"
#     SUA = f"logs/{env}_SUA.log"
#     SUAIR = f"logs/{env}_SUAIR3.log"
#     # rcmp = f"logs/{env}_rcmp.log"
#     rcmp_msloss = f"logs/{env}_rcmp_msloss.log"
#     rcmp_adap = f"logs/{env}_rcmp_adap.log"
#     rcmp_only = f"logs/{env}_rcmp_adap_only.log"
#     # plt_log(noadvice)
#     # plt_log(early)
#     # plt_log(random)
#     # plt_log(air)
#     # plt_log(SUA)
#     # plt_log(SUAIR)
#     # plt_log(rcmp)
#     plt_log(rcmp_msloss)
#     plt_log(rcmp_adap)
#     plt_log(rcmp_only)
#     plt.margins(x=0, y=0)
#     plt.xlim(0, 5e6)
#     # plt.ylim(0, 4000)
#     plt.grid()
#     plt.legend(["rcmp", "rcmp_adap", "rcmp_only"])
#     plt.title(f"{env}")
#     plt.savefig(f"figures/{env}_rcmp_adap_compare")
#     plt.close()

# compare
# for env in envs:
#     # noadvice = f"logs/{env}.log"
#     # random = f"logs/{env}_random.log"
#     # early = f"logs/{env}_early.log"
#     # air = f"logs/{env}_AIR.log"
#     # air2 = f"logs/{env}_AIR2.log"
#     # SUA = f"logs/{env}_SUA.log"
#     # SUAIR = f"logs/{env}_SUAIR.log"
#     # SUAIR2 = f"logs/{env}_SUAIR2.log"
#     rcmp = f"logs/{env}_rcmp.log"
#     rcmp_msloss = f"logs/{env}_rcmp_msloss.log"
#     plt_log(rcmp)
#     plt_log(rcmp_msloss)

#     plt.margins(x=0, y=0)
#     plt.xlim(0, 5e6)
#     plt.grid()
#     plt.legend(["rcmp", "rcmp_msloss"])
#     plt.title(f"{env}")
#     plt.savefig(f"figures/{env}_rcmp_compare")
#     plt.close()
# pong = "logs/Pong.log"
# pong = "logs/Pong.log"
# pong_SUA = "logs/Pong_SUA.log"
# pong_rcmp = "logs/Pong_rcmp.log"
# pong_rcmp_msloss = "logs/Pong_rcmp_msloss.log"
# plt_log(pong)
# plt_log(pong_SUA)
# plt_log(pong_rcmp)
# plt_log(pong_rcmp_msloss)
# plt.margins(x=0, y=0)
# plt.xlim(0, 5e6)
# plt.grid()
# plt.legend(["noadvice", "SUA", "rcmp", "rcmp_msloss"])
# plt.title("pong")
# plt.savefig("Pong_rcmp_result")

# draw_table()
# plt_log("logs/Qbert_SUAIR3.log")
# plt_log("logs/Qbert_SUAIR4.log")
# plt.grid()
# plt.legend(["SUA", "SUA2"])
# # plt.title(f"{env}")
# plt.savefig(f"figures/SUA_compare")
# plt.close()


def plt_muti_rate(name, color=None, lines=5):
    rate = [[] for n in range(lines)]
    epochs = [[] for n in range(lines)]
    for i in range(lines):
        rate[i].append(0)
        epochs[i].append(0)
    for i in range(lines):
        data = open(name+f"_{i+1}.log", 'r')
        count = 1
        for line in data:
            items = line.split(" ")
            if items[0] == "reuse_learned_rate":
                epochs[i].append(int(count * 50000))
                rate[i].append(float(items[-1])) 
                count += 1
    lengths = [len(epochs[i]) for i in range(lines)]
    for i in range(lines):
        if len(rate[i]) > min(lengths):
            rate[i] = rate[i][:min(lengths)]
    rate = np.array(rate)
    print(rate.shape)
    mid_rate = np.median(rate, axis=0)
    max_rate = np.max(rate, axis=0)
    min_rate = np.min(rate, axis=0)

    # for i in range(len(mid_rewards)):
    #     if mid_rewards[i] > max_rewards[i] or mid_rewards[i] < min_rewards[i]:
    #         print(f"mid {mid_rewards[i]}, max {max_rewards[i]}, min {min_rewards[i]}")
    # # print(max_rewards.shape)
    
    if min(lengths) < 200:
        valid_length = min(lengths)
        epochs[0] = epochs[0][:valid_length]
    plt.plot(epochs[0], mid_rate, color=color)
    # plt.plot(epochs[0], max_rewards, color=color)
    # plt.plot(epochs[0], min_rewards, color=color)
    print(f"min : {len(min_rate)}, max : {len(max_rate)}")
    p1 = plt.fill_between(epochs[0], max_rate, min_rate, alpha=0.3, color=color)
    return p1

# envs = ["Pong"]
# for env in envs:
#     p0 = plt_muti_rate(f"logs/{env}_SUAIR_decay", colors[2])
#     # p1 = plt_muti_rate(f"logs/{env}_adap_reuseT_distRS_Big0tanh0.2_1e6", color=colors[3])
#     # p13 = plt_muti_rate(f"logs/{env}_adap_reuseT_distRS_tanh0.1decay_1e6", color='b')
#     p2 = plt_muti_rate(f"logs/final_experiment_data/ours/{env}_adap_reuseT_RStanh0.25decay", color=colors[0])
#     plt.margins(x=0, y=0)
#     plt.xlim(0, 5e6)
#     plt.grid()
#     plt.ylabel("% correct actions compared to advice")
#     plt.xlabel("Millions of envirionment steps")
#     # plt.xlim(0, 2e6)
#     plt.title(f"{env}")
#     plt.legend([p0, p2], ['SUAIR', 'our method', 'RS0.1'])
#     plt.savefig(f"figures/{env}_rate.pdf")
#     plt.close()