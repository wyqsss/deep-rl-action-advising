import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def plt_log(logfile):
    data = open(logfile, 'r')
    reward = []
    epoch = []
    for line in data:
        items = line.split(" ")
        if items[0] == "Evaluation" and items[1] == '@':
            epoch.append(int(float(items[2])))
            reward.append(float(items[-1]))
    plt.plot(epoch, gaussian_filter1d(reward, sigma=3))

    print(f"epoch : {epoch[-1]}, reward : {reward[-1]}")


envs = ["Enduro", "Freeway", "Pong", "Qbert", "Seaquest"]
for env in envs:
    noadvice = f"logs/{env}.log"
    random = f"logs/{env}_random.log"
    early = f"logs/{env}_early.log"
    air = f"logs/{env}_AIR2.log"
    SUA = f"logs/{env}_SUA.log"
    SUAIR = f"logs/{env}_SUAIR3.log"
    # rcmp = f"logs/{env}_rcmp.log"
    rcmp_msloss = f"logs/{env}_rcmp_msloss.log"
    plt_log(noadvice)
    plt_log(random)
    plt_log(early)
    plt_log(air)
    plt_log(SUA)
    plt_log(SUAIR)
    # plt_log(rcmp)
    plt_log(rcmp_msloss)
    plt.margins(x=0, y=0)
    plt.xlim(0, 5e6)
    plt.grid()
    plt.legend(["NA", "EA", "RA", "AIR", "SUA", "SUA-AIR", "rcmp", "rcmp_msloss"])
    plt.title(f"{env}")
    plt.savefig(f"figures/{env}_result")
    plt.close()

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