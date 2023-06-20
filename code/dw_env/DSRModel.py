#!/usr/bin/env python
# -*-coding:utf-8 -*-


import pandapower as pp
import pandapower.topology as tp
import numpy as np
import copy as cp
import networkx as nx


class DSRModel:

    def __init__(self):
        self.net = pp.from_excel("dw_env/case/IEEE-123.xlsx")
        pp.runpp(self.net)
        # 读取初始开关状态
        self.sw_state_base = self.net.switch.closed.values.copy()
        # 常闭/常开开关
        self.sw_always_open = np.where(self.sw_state_base == True)[0]
        self.sw_always_close = np.where(self.sw_state_base == False)[0]

        self.line_in_service_base = self.net.line.in_service
        # 读取初始节点有功无功
        self.p_load_base = cp.copy(self.net.load.p_mw.values.copy())
        self.q_load_base = cp.copy(self.net.load.q_mvar.values.copy())
        # 定义节点电压上下限
        self.vm_min = np.ones(self.get_bus_num()) * 0.9
        self.vm_max = np.ones(self.get_bus_num()) * 1.1
        # TODO 定义支路功率上下限
        self.p_max = abs(self.net.res_line.p_from_mw.values.copy()) * 1.5
        self.p_min = -abs(self.net.res_line.p_from_mw.values.copy()) * 1.5
        # 初始化评价指标参数
        self.p_loss_base = 0
        self.sw_fault = 0
        self.action_count = 0
        # 初始化潮流
        pp.runpp(self.net)

    def reset(self):
        # 重置开关状态
        self.net.switch.closed = self.sw_state_base

        # 施加线路故障
        self.net.line.in_service = self.line_in_service_base
        self.net.line.in_service[np.random.randint(0, len(self.net.line), size=1)] = False
        while 0 in np.asarray(nx.Graph(tp.create_nxgraph(self.net)).degree)[:, 1]:
            self.net.line.in_service = self.line_in_service_base
            self.net.line.in_service[np.random.randint(0, len(self.net.line), size=1)] = False
        # 节点有功无功随机波动
        # random_array = (np.random.random(self.p_load_base.shape[0]) * 2 - 1) * 0.2 + 1  # 生成[0.8~1.2]的随机数组
        # self.net.load.p_mw = self.p_load_base * random_array
        # self.net.load.q_mvar = self.q_load_base * random_array
        # 施加随机故障
        self.sw_fault = np.random.choice(self.sw_always_open, 1)[0]
        self.net.switch.closed[self.sw_fault] = False
        # 潮流计算初始化
        pp.runpp(self.net)
        # 重置状态量
        self.p_loss_base = self.get_power_unsupplied()
        self.action_count = 0

    def action(self, sw_open, sw_close):
        # 执行一组操作，打开一个开关，同时闭合一个开关
        self.net.switch.closed[sw_open] = False
        self.net.switch.closed[sw_close] = True
        self.action_count += 1
        pp.runpp(self.net)

    def action_open(self, sw_open):
        self.net.switch.closed[sw_open] = False
        self.action_count += 1
        pp.runpp(self.net)

    def action_close(self, sw_close):
        self.net.switch.closed[sw_close] = True
        self.action_count += 1
        pp.runpp(self.net)

    def get_bus_num(self):
        return self.net.bus.shape[0]

    def get_line_num(self):
        return self.net.line.shape[0]

    def get_sw_num(self):
        return self.net.switch.shape[0]

    def get_bus_state(self):
        # 各列分别为节点状态量：电压幅值，电压相角
        return np.r_[self.net.res_bus.vm_pu.values.copy(),
                     self.net.res_bus.va_degree.values.copy()]

    def get_line_state(self):
        # 各列分别为支路状态量：有功功率，无功功率
        return np.r_[self.net.res_line.p_from_mw.values.copy(),
                     self.net.res_line.q_from_mvar.values.copy()]

    def get_sw_state(self):
        return self.net.switch.closed.values.copy()

    def get_fault(self):
        return self.sw_fault

    def get_loop(self):
        # 先转换为networkx无向图，再来搜索闭环
        graph = nx.Graph(tp.create_nxgraph(self.net))
        return nx.cycle_basis(graph)

    # 网损
    def get_power_loss(self):
        return np.sum(self.net.res_line.pl_mw)

    # 失负荷量
    def get_power_unsupplied(self):
        # 获取失电母线
        unsupplied_buses = tp.unsupplied_buses(self.net)
        # 计算失电量
        result = 0
        for nbus in unsupplied_buses:
            result += self.net.load.loc[self.net.load.bus == nbus, 'p_mw'].values[0]
        return result

    # 恢复负荷量
    def get_power_recovered(self):
        return self.p_loss_base - self.get_power_unsupplied()

    # 电压越限量
    def get_over_voltage(self):
        vm_pu = self.net.res_bus.vm_pu.values.copy()
        # 寻找电压越限节点
        idx0 = (vm_pu > self.vm_max)
        idx1 = (vm_pu < self.vm_min)
        # 返回累计偏差值
        return np.sum(vm_pu[idx0] - self.vm_max[idx0]) + np.sum(self.vm_min[idx1] - vm_pu[idx1])

    # 功率越限量
    def get_over_load(self):
        p_mw = self.net.res_line.p_from_mw.values.copy()
        # 寻找有功越限支路
        idx0 = (p_mw > self.p_max)
        idx1 = (p_mw < self.p_min)
        # 返回累计偏差值
        return np.sum(p_mw[idx0] - self.p_max[idx0]) + np.sum(self.p_min[idx1] - p_mw[idx1])

    def get_action_count(self):
        return self.action_count

    def is_recovered(self):
        return self.get_power_unsupplied() == 0


if __name__ == '__main__':
    # 建立模型
    model = DSRModel()
    # 模型施加随机故障
    model.reset()
    print('Fault Switch ID: ', model.get_fault())
    print('Power Loss: ', model.get_power_unsupplied())
    print('Over Voltage: ', model.get_over_voltage())
    while True:
        sw_id = int(input("Close Switch: "))
        model.action(model.get_fault(), sw_id)
        print('Power Loss: ', model.get_power_unsupplied())
        print('Power Recovered: ', model.get_power_recovered())
        print('Over Voltage: ', model.get_over_voltage())
        print('EM Loop: ', model.get_loop())
        print('Action Count: ', model.get_action_count())
