#!/usr/bin/env python
# -*-coding:utf-8 -*-

import math
import numpy as np

from gym import spaces
from dw_env.DSRModel import DSRModel

MAX_ACTION_COUNT = 10  # 最大动作次数

C0 = 5  # 软约束惩罚
C1 = 10  # 硬约束惩罚

# 偏置量
ALPHA = 0.03
BETA = 0.05
GAMMA = 0.07


class DSREnv:

    def __init__(self):
        # 初始化本地配电网模型
        self.model = DSRModel()
        self.state = None
        # 初始化环境状态
        self.model.reset()
        # 定义电压幅值状态空间上下限
        self.vm_min = self.model.vm_min
        self.vm_max = self.model.vm_max
        # 定义电压相角状态空间上下限
        self.va_min = np.ones(self.model.get_bus_num()) * -math.pi
        self.va_max = np.ones(self.model.get_bus_num()) * math.pi
        # 定义支路有功状态空间上下限
        self.p_min = np.ones(self.model.get_line_num()) * -1.5
        self.p_max = np.ones(self.model.get_line_num()) * 1.5
        # 定义支路无功状态空间上下限
        self.q_min = np.ones(self.model.get_line_num()) * -100
        self.q_max = np.ones(self.model.get_line_num()) * 100
        # 定义开关状态上下限
        self.sw_min = np.ones(self.model.get_sw_num()) * 0
        self.sw_max = np.ones(self.model.get_sw_num()) * 1
        # 定义故障开关上下限
        self.fault_min = 0
        self.fault_max = self.model.get_sw_num()
        # 状态空间定义：故障开关，上一次动作开关，节点电压幅值，节点电压相角，支路有功功率，支路无功功率，当前开关状态
        observation_space_lb = np.float32(np.r_[self.fault_min, self.fault_min,
        self.vm_min, self.va_min, self.p_min, self.q_min, self.sw_min])
        observation_space_ub = np.float32(np.r_[self.fault_max, self.fault_max,
        self.vm_max, self.va_max, self.p_max, self.q_max, self.sw_max])
        # 定义gym动作空间，第一个动作为开断，第二个动作为闭合
        self.action_space = spaces.Discrete(self.model.get_sw_num())
        # 定义gym观测空间
        self.observation_space = spaces.Box(observation_space_lb, observation_space_ub, dtype=np.float32)

        # 上一次动作
        self.last_action = -1
        # 上一次动作后的失负荷量
        self.last_power_unsupplied = -1

    def step(self, action):
        # 开关状态翻转
        sw_state = self.model.get_sw_state()
        if sw_state[action]:
            self.model.action_open(action)
        else:
            self.model.action_close(action)

        # 读取更新状态
        state = self.get_state()

        # 计算奖励值
        reward = (self.model.get_power_recovered() + ALPHA) * \
                 (1 / (self.model.get_action_count() + BETA)) * \
                 (1 / (self.model.get_power_loss() + GAMMA))

        reward = -2 if (self.model.get_power_unsupplied() > self.last_power_unsupplied) else reward

        done = bool(
            # 终止条件1：完成供电恢复
            (self.model.get_power_unsupplied() == 0 and len(self.model.get_loop()) == 0)
            # 终止条件2：电压越限
            or (self.model.get_over_voltage() > 0)
            # 终止条件3：功率越限 (数据缺失，暂时忽略)
            # or (self.model.get_over_load() > 0)
            # 终止条件5：动作次数越限
            or (self.model.get_action_count() > MAX_ACTION_COUNT)
            # 终止条件6：连续重复动作
            or (self.last_action == action)
            # 终止条件7：故障开关动作
            or (self.model.get_fault() == action)
            # 终止条件8：失负荷量增加
            # or (self.model.get_power_unsupplied() > self.last_power_unsupplied)
        )

        # 惩罚1：计算电压越限惩罚
        reward -= (self.model.get_over_voltage() * C0)
        # 惩罚2：计算功率越限惩罚 (数据缺失，暂时忽略)
        # reward -= self.model.get_over_load() * C0
        # 惩罚3：计算环路惩罚
        reward -= (C0 if (len(self.model.get_loop()) > 0) else 0)
        # 惩罚4：连续重复动作惩罚
        reward -= (C1 if (self.last_action == action) else 0)
        # 惩罚5：故障开关动作惩罚
        reward -= (C1 if (self.model.get_fault() == action) else 0)
        # 惩罚6：失负荷量增加
        reward -= (C1 if (self.model.get_power_unsupplied() > self.last_power_unsupplied) else 0)

        if done:
            if self.model.get_power_unsupplied() == 0 and len(self.model.get_loop()) == 0:
                print('完成供电恢复')
                reward += 20
            else:
                reward -= 10
                if self.model.get_over_voltage() > 0:
                    print('电压越限')
                # elif self.model.get_over_load() > 0:
                #     print('功率越限 (数据缺失，暂时忽略)')
                elif self.model.get_action_count() > MAX_ACTION_COUNT:
                    print('动作次数越限')
                elif self.last_action == action:
                    print('连续重复动作')
                elif self.model.get_fault() == action:
                    print('故障开关动作')
                elif self.model.get_power_unsupplied() > self.last_power_unsupplied:
                    print('失负荷量增加')
                else:
                    print('未知终止条件')
        # 更新动作后状态
        self.last_action = action
        self.last_power_unsupplied = self.model.get_power_unsupplied()

        return np.array(state, dtype=np.float32), reward, done, False, {}

    def reset(self, test=False, test_id=None):
        self.model.reset(test, test_id)
        self.last_action = self.model.get_fault()
        self.last_power_unsupplied = self.model.get_power_unsupplied()
        # 调用状态函数，返回新的状态
        return np.array(self.get_state(), dtype=np.float32), {}

    def get_state(self):
        result = np.r_[self.model.get_fault(),  # 故障开关
        self.last_action,  # 上一次动作开关
        self.model.get_bus_state(),  # 节点电压幅值，节点电压相角
        self.model.get_line_state(),  # 支路有功功率，支路无功功率
        self.model.get_sw_state().astype(int)]  # 当前开关状态
        # 失电节点电压暂定为1，避免产生电压越限惩罚
        result[np.isnan(result)] = 1

        return result
