from crypt import methods
from turtle import ScrolledCanvas
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from tabulate import tabulate
import seaborn as sns
from pandas import DataFrame
import numpy as np
import pandas as pd

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

def plt_muti_log(name, color=None, lines=5):
    rewards = [[] for n in range(lines)]
    epochs = [[] for n in range(lines)]
    for i in range(lines):
        data = open(name+f"_{i+1}.log", 'r')
        for line in data:
            items = line.split(" ")
            if items[0] == "Evaluation" and items[1] == '@':
                epochs[i-1].append(int(float(items[2])))
                rewards[i-1].append(float(items[-1])) 
    lengths = [len(epochs[i]) for i in range(lines)]
    for i in range(lines):
        if len(rewards[i]) > min(lengths):
            rewards[i] = rewards[i][:min(lengths)]
    rewards = np.array(rewards)
    print(rewards.shape)
    mid_rewards = np.median(rewards, axis=0)
    max_rewards = np.max(rewards, axis=0)
    min_rewards = np.min(rewards, axis=0)

    # for i in range(len(mid_rewards)):
    #     if mid_rewards[i] > max_rewards[i] or mid_rewards[i] < min_rewards[i]:
    #         print(f"mid {mid_rewards[i]}, max {max_rewards[i]}, min {min_rewards[i]}")
    # # print(max_rewards.shape)
    
    if min(lengths) < 200:
        valid_length = min(lengths)
        epochs[0] = epochs[0][:valid_length]
    plt.plot(epochs[0],gaussian_filter1d(mid_rewards, sigma=1), color=color)
    # plt.plot(epochs[0], max_rewards, color=color)
    # plt.plot(epochs[0], min_rewards, color=color)
    print(f"min : {len(min_rewards)}, max : {len(max_rewards)}")
    p1 = plt.fill_between(epochs[0], max_rewards, min_rewards, alpha=0.3, color=color)
    return p1
# envs = ["Pong"]
# for env in envs:
#     p0 = plt_muti_log(f"/mnt/nfs/wyq/{env}/{env}_SUAIR", colors[2])
#     # p1 = plt_muti_log(f"logs/{env}_adap_acbyol_100epoch_clearbuf_true", color=colors[0])
#     # p2 = plt_muti_log(f"logs/{env}_adap_acbyol_100epoch_onetrain_true", colors[1])
#     # p3 = plt_muti_log(f"logs/{env}_adap_acbyol_100epoch_true", colors[2])
#     # p4 = plt_muti_log(f"/mnt/nfs/wyq/{env}/{env}_adap_100epoch_true", colors[3])
#     # p5 = plt_muti_log(f"logs/{env}_adap_acbyol_20epoch_newnet_clearbuf_true", color='cyan')
#     p5 = plt_muti_log(f"/mnt/nfs/wyq/{env}/{env}_adap_acbyol_20epoch_newnet_clearbuf_true", color='cyan')
#     p6 = plt_muti_log(f"/mnt/nfs/wyq/{env}/{env}_adap_acbyol_20epoch_newnet_reuse", color=colors[0])
#     p7 = plt_muti_log(f"logs/{env}_adap_acbyol_reuse_RS0.2", color=colors[1])
#     p8 = plt_muti_log(f"logs/{env}_adap_acbyol_reuse_RS_lateeps", color=colors[3])
#     p9 = plt_muti_log(f"logs/{env}_adap_reuse_RS_pos_stateneg", color='b')
#     p10 = plt_muti_log(f"logs/{env}_adap_reuse_RS_600ep", color='y')
#     plt.margins(x=0, y=0)
#     plt.xlim(0, 5e6)
#     plt.grid()
#     # plt.xlim(0, 5e6)
#     plt.title(f"{env}")
#     plt.legend([p0, p5, p6,p7, p8, p9, p10], ['SUAIR', 'adap_dist', "adap_dist_reuse", "reuse_RS", 'lateeps', 'stateneg', 'ep_decay'])
#     plt.savefig(f"figures/{env}_muti_comparenet_5e6")
#     plt.close()
envs = ["Freeway"]
for env in envs:
    p0 = plt_muti_log(f"/nfs3-p1/wyq/gpu03/logs/{env}_SUAIR", colors[2])
    # p1 = plt_muti_log(f"logs/{env}_adap_acbyol_100epoch_clearbuf_true", color=colors[0])
    # p2 = plt_muti_log(f"logs/{env}_adap_acbyol_100epoch_onetrain_true", colors[1])
    # p3 = plt_muti_log(f"logs/{env}_adap_acbyol_100epoch_true", colors[2])
    # p4 = plt_muti_log(f"/mnt/nfs/wyq/{env}/{env}_adap_100epoch_true", colors[3])
    # p5 = plt_muti_log(f"logs/{env}_adap_acbyol_20epoch_newnet_clearbuf_true", color='cyan')
    p5 = plt_muti_log(f"logs/{env}_adap_reuse_distRS0.5_zetaprob", color='cyan')
    p6 = plt_muti_log(f"/nfs3-p1/wyq/gpu04/logs/{env}_adap_acbyol_reuse_RS0.5_decay", color=colors[0])
    # p7 = plt_muti_log(f"logs/{env}_adap_reuse_RS_pos0.5_neg0.1", color=colors[1])
    # p8 = plt_muti_log(f"logs/{env}_adap_reuse_RS_ucthrold", color=colors[3])
    # p9 = plt_muti_log(f"logs/{env}_adap_reuse_RS_obsnextuc", color=colors[0])
    # p10 = plt_muti_log(f"logs/{env}_adap_reuse_RS0.5", color=colors[0])
    # p11 = plt_muti_log(f"logs/{env}_adap_reuse_RS_zetadecay1000_reuseall", color=colors[1], lines=4)
    # p12 = plt_muti_log(f"logs/{env}_adap_reuse_RS_zetadecay2000_reuseall", color=colors[3])
    # p13 = plt_muti_log(f"logs/{env}_adap_reuse_distRS", color='b')
    # p14 = plt_muti_log(f"logs/{env}_adap_reuse_distRS_bufferneg", color='c')
    plt.margins(x=0, y=0)
    plt.xlim(0, 5e6)
    plt.grid()
    # plt.ylim(0, 4500)
    plt.title(f"{env}")
    plt.legend([p0, p5, p6], ['SUAIR', 'adap_dist6e5', "rs_0.5", 'dist_RS', 'distRS_neg', 'zeta2000_reuseall_usesubmodel', 'zeta2000_reuse0.5_usesubmodel'])
    plt.savefig(f"figures/{env}_muti_test_5e6")
    plt.close()
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