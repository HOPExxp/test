# -*-coding:utf-8-*-
import os
from fmpy import *
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class CoolingSystemSimulator:
    def __init__(self, fmu_path):
        self.fmu_path = fmu_path
        self.model_info = read_model_description(fmu_path, validate=False)
        self.results = None
        self.input_signals = None  # 新增：存储输入信号

        # 获取所有输入输出变量信息
        self.input_vars = [v for v in self.model_info.modelVariables if v.causality == 'input']
        self.output_vars = [v for v in self.model_info.modelVariables if v.causality == 'output']

        # 找出所有功率相关的变量
        self.power_vars = [v.name for v in self.model_info.modelVariables
                           if v.causality != 'input' and v.type == 'Real'
                           and v.name.lower().endswith('.p') and v.unit == "W"]

        print(f"找到 {len(self.power_vars)} 个功率变量: {self.power_vars[:5]}...")  # 只显示前5个

    def create_input_signals(self, start_time, stop_time, step_size, input_values=None, input_csv_path=None):
        """
        为所有输入变量创建输入信号，支持从CSV文件读取时间序列数据

        Parameters:
        start_time: 仿真开始时间
        stop_time: 仿真结束时间
        step_size: 时间步长
        input_values: 可选的输入值字典，用于CSV中没有的变量或覆盖CSV中的值
        input_csv_path: CSV文件路径，包含时间序列输入数据
        """
        from scipy.interpolate import interp1d
        import pandas as pd
        import os

        # 创建时间序列
        time_points = np.linspace(start_time, stop_time, int((stop_time - start_time) / step_size) + 1)

        # 默认输入值（可以根据需要修改）
        default_values = {
            'coolingLoad': 7000000.0,  # 冷却负荷
            'n': 4,  # 运行的制冷机数量
            'Mode': 1,  # 设备运行模式信号
            'CT_c': 1,  # 冷却塔变频信号
            'CHP_c': 1,  # 冷冻水泵控制信号
            'CP_c': 1,  # 冷却水泵控制信号
            'Chiller_c': True,  # 冷水机组开关信号
            'Tset': 273.15 + 17,  # 冷水机组出水温度设定值
            'Twb': 273.15 + 27,  # 室外空气湿球温度
            'valveOpen': 0,  # 阀门开度控制
            'Val9': 0,  # 储罐阀门9控制
            'Val10': 0,  # 储罐阀门10控制
            'StoragePumpSignal': 0  # 储罐泵控制信号
        }

        # 如果提供了自定义值，更新默认值
        if input_values:
            default_values.update(input_values)

        # 从CSV文件读取输入数据（如果提供了路径）
        csv_data = {}
        if input_csv_path and os.path.exists(input_csv_path):
            try:
                df = pd.read_csv(input_csv_path)
                # 确保有时间列
                if 'time' in df.columns:
                    # 为每个变量创建插值函数
                    for var in self.input_vars:
                        var_name = var.name
                        if var_name in df.columns:
                            # 创建插值函数
                            interp_func = interp1d(
                                df['time'],
                                df[var_name],
                                kind='linear',
                                bounds_error=False,
                                fill_value=(df[var_name].iloc[0], df[var_name].iloc[-1])
                            )
                            # 应用插值
                            csv_data[var_name] = interp_func(time_points)
                            print(f"从CSV文件读取变量 {var_name} 的数据")
                else:
                    print("警告: CSV文件中没有'time'列，无法读取时间序列数据")
            except Exception as e:
                print(f"读取CSV文件时出错: {e}")

        # 创建输入信号数据结构
        input_signals = []
        dtype_list = [('time', np.float64)]

        for var in self.input_vars:
            var_name = var.name
            # 优先使用CSV中的数据，然后是自定义值，最后是默认值
            if var_name in csv_data:
                values = csv_data[var_name]
            elif input_values and var_name in input_values:
                value = input_values[var_name]
                values = np.full_like(time_points, value)
            elif var_name in default_values:
                value = default_values[var_name]
                values = np.full_like(time_points, value)
            else:
                # 如果没有提供值，使用0或False作为默认值
                if var.type == 'Boolean':
                    values = np.full_like(time_points, False, dtype=np.bool_)
                elif var.type == 'Integer':
                    values = np.full_like(time_points, 0, dtype=np.int32)
                else:
                    values = np.full_like(time_points, 0.0, dtype=np.float64)

            # 根据变量类型处理数据
            if var.type == 'Real':
                input_signals.append(values.astype(np.float64))
                dtype_list.append((var_name, np.float64))
            elif var.type == 'Integer':
                input_signals.append(values.astype(np.int32))
                dtype_list.append((var_name, np.int32))
            elif var.type == 'Boolean':
                input_signals.append(values.astype(np.bool_))
                dtype_list.append((var_name, np.bool_))

        # 创建结构化数组
        structured_array = np.zeros(len(time_points), dtype=dtype_list)
        structured_array['time'] = time_points

        for i, var in enumerate(self.input_vars):
            var_name = var.name
            structured_array[var_name] = input_signals[i]

        # 保存输入信号供后续导出
        self.input_signals = structured_array

        return structured_array

    def export_input_signals(self, filename='input_signals.csv'):
        """
        导出输入信号到CSV文件

        Parameters:
        filename: 保存输入信号的CSV文件名
        """
        if self.input_signals is None:
            print("没有输入信号可导出，请先运行仿真或创建输入信号")
            return

        try:
            # 使用fmpy的write_csv函数保存结构化数组
            write_csv(filename, self.input_signals)
            print(f'输入信号已保存到 {os.path.abspath(filename)}')
        except Exception as e:
            print(f"导出输入信号时出错: {e}")

    def simulate(self, start_time=0.0, stop_time=7200.0, step_size=5.0,
                 input_values=None, output_variables=None, input_csv_path=None):
        """
        运行仿真

        Parameters:
        start_time: 仿真开始时间
        stop_time: 仿真结束时间
        step_size: 时间步长
        input_values: 可选的输入值字典
        output_variables: 可选的输出变量列表，如果为None则包含所有功率变量和基本输出
        input_csv_path: CSV文件路径，包含时间序列输入数据
        """
        # 设置默认输出变量
        if output_variables is None:
            output_variables = ['supplyTemp', 'returnTemp'] + self.power_vars

        # 创建输入信号
        input_signals = self.create_input_signals(
            start_time, stop_time, step_size,
            input_values=input_values,
            input_csv_path=input_csv_path
        )

        # 运行仿真
        try:
            self.results = simulate_fmu(
                self.fmu_path,
                start_time=start_time,
                stop_time=stop_time,
                output_interval=step_size,
                output=output_variables,
                input=input_signals,
                validate=False
            )

            # 计算总能耗（对功率积分）
            self.calculate_energy_consumption()

            return True
        except Exception as e:
            print(f"仿真过程中发生错误: {e}")
            return False

    def calculate_energy_consumption(self):
        """计算总能耗（对功率积分）"""
        if self.results is None:
            print("没有仿真结果可用")
            return

        # 找出所有功率变量
        power_columns = [col for col in self.results.dtype.names if col in self.power_vars]

        if not power_columns:
            print("没有找到功率变量")
            return

        # 计算总功率（所有功率变量之和）
        total_power = np.zeros_like(self.results['time'])
        for col in power_columns:
            total_power += self.results[col]

        # 对功率积分得到能耗（使用梯形法积分）
        time_diff = np.diff(self.results['time'])
        energy = np.zeros_like(total_power)

        for i in range(1, len(energy)):
            energy[i] = energy[i - 1] + 0.5 * (total_power[i - 1] + total_power[i]) * time_diff[i - 1]

        # 将结果添加到结果数组中
        # 需要创建一个新的结构化数组来包含能耗数据
        new_dtype = self.results.dtype.descr + [('total_power', np.float64), ('total_energy', np.float64)]
        new_results = np.zeros(len(self.results), dtype=new_dtype)

        for name in self.results.dtype.names:
            new_results[name] = self.results[name]

        new_results['total_power'] = total_power
        new_results['total_energy'] = energy

        self.results = new_results
        self.total_energy = energy[-1]  # 总能耗（最终值）

        print(f"总能耗: {self.total_energy/3.6e6:.2f} kWh")  # 转换为kWh

    def save_results(self, filename='result.csv'):
        """保存结果到CSV文件"""
        if self.results is not None:
            write_csv(filename, self.results)
            print(f'结果已保存到 {os.path.abspath(filename)}')
        else:
            print("没有结果可保存")

    def plot_results(self, variables=None, save_path=None):
        """
        绘制结果图表

        Parameters:
        variables: 要绘制的变量列表，如果为None则绘制默认变量
        save_path: 图片保存路径，如果为None则不保存
        """
        if self.results is None:
            print("没有结果可绘制")
            return

        if variables is None:
            variables = ['supplyTemp', 'returnTemp', 'total_power', 'total_energy']

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, var in enumerate(variables):
            if i >= len(axes):
                break

            if var in self.results.dtype.names:
                # 特殊处理温度和能耗单位
                if var in ['supplyTemp', 'returnTemp']:
                    data = self.results[var] - 273.15  # 转换为摄氏度
                    unit = "°C"
                elif var == 'total_energy':
                    data = self.results[var] / 3.6e6  # 转换为kWh
                    unit = "kWh"
                elif var == 'total_power':
                    data = self.results[var] / 1000  # 转换为kW
                    unit = "kW"
                else:
                    data = self.results[var]
                    unit = ""

                axes[i].plot(self.results['time'], data)
                axes[i].set_xlabel('Time [s]')
                axes[i].set_ylabel(f'{var} [{unit}]')
                axes[i].set_title(var)
                axes[i].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"图表已保存到 {save_path}")

        plt.show()

    def get_input_variables(self):
        """返回所有输入变量的信息"""
        return [(v.name, v.type, v.causality) for v in self.input_vars]

    def get_output_variables(self):
        """返回所有输出变量的信息"""
        return [(v.name, v.type, v.causality) for v in self.output_vars]


# 使用示例
if __name__ == "__main__":
    # 1. 初始化仿真器
    FMU_PATH = r"C:\Users\xxp\Documents\Dymola\CET-zhuzhouyidong\simulation_changeinput\CoolingStation_0changeinput_System_Coolingstation.fmu"
    simulator = CoolingSystemSimulator(FMU_PATH)

    # 2. 查看可用的输入输出变量
    print("输入变量:")
    for name, type_, causality in simulator.get_input_variables():
        print(f"  {name} ({type_}, {causality})")

    print("\n输出变量:")
    for name, type_, causality in simulator.get_output_variables():
        print(f"  {name} ({type_}, {causality})")

    # # 3. 设置自定义输入值（可选）
    # custom_inputs = {
    #     'coolingLoad': 8000000.0,  # 增加冷却负荷
    #     'Tset': 273.15 + 16,  # 降低设定温度
    #     'n': 3,  # 减少运行的制冷机数量
    #     # 可以添加更多自定义输入...
    # }

    # 4. 运行仿真
    success = simulator.simulate(
        start_time=0.0,
        stop_time=7200.0,
        step_size=5.0,
        input_csv_path='input_data.csv'
    )

    if success:
        # 5. 保存结果
        simulator.save_results('cooling_system_results.csv')
        simulator.export_input_signals('input_signals.csv')  # 导出输入信号

        # 6. 绘制结果
        simulator.plot_results(save_path='cooling_system_plots.png')
        plt.figure(figsize=(10, 6))
        plt.plot(simulator.results['time'], simulator.results['supplyTemp'] - 273.15, label='Supply Temperature [°C]',
                 color='tab:blue')
        plt.plot(simulator.results['time'], simulator.results['returnTemp'] - 273.15, label='Return Temperature [°C]',
                 color='tab:red')

        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [°C]')
        plt.title('Cooling System Simulation Results')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('fmpy_results.png', dpi=150)
        plt.show()

        # 7. 打印能耗信息
        print(f"\n仿真完成，总能耗: {simulator.total_energy/3.6e6:.2f} kWh")