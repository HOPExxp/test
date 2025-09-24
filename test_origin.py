#-*-coding:utf-8-*-
import os
from fmpy import *
from fmpy.util import plot_result
import matplotlib.pyplot as plt

from collections import Counter
import numpy as np

# 1. 文件路径（改这里！）
# FMU_PATH = r"C:\Users\xxp\Documents\Dymola\CET-zhuzhouyidong\simulation_changeinput\CoolingStation_0changeinput_System_Coolingstation.fmu"
FMU_PATH = r"C:\Users\xxp\Documents\Dymola\CET-zhuzhouyidong\simulation\CoolingStation_System_Coolingstation.fmu"

# 2. 读模型描述（跳过 XML 校验）
info = read_model_description(
    FMU_PATH,
    validate=False
)
print('FMI version:', info.fmiVersion)
print('Variables  :', len(info.modelVariables))

# 3. 基本参数
START_TIME = 0.0        # 仿真开始时间 [s]
STOP_TIME  = 7200.0      # 仿真结束时间 [s]
STEP_SIZE  = 1.0        # 通信步长 [s]

# 4. 创建正确格式的输入信号
# 创建时间序列（更密集的点以获得更好的插值效果）
time_points = np.linspace(0, STOP_TIME, int(STOP_TIME/STEP_SIZE) + 1)

# 创建恒定冷却负荷（7000000 W）
cooling_load_values = np.full_like(time_points, 7000000.0)

# 创建结构化数组，这是FMPy需要的格式
input_signals = np.array(
    list(zip(time_points, cooling_load_values)),
    dtype=[('time', np.float64), ('coolingLoad', np.float64)]
)

print("输入信号格式示例:")
print(input_signals[:5])  # 打印前5个点

# 5. 检查模型变量
print("\n模型输入变量:")
inputs = [v.name for v in info.modelVariables if v.causality == 'input']
print(inputs)

print("\n模型输出变量:")
outputs = [v.name for v in info.modelVariables if v.causality == 'output']
print(outputs)

#5.0 统计 causality 及其频数
causal_cnt = Counter(v.causality for v in info.modelVariables)
print('causality 频数表：')
for k, v in causal_cnt.items():
    print(f'{k or "None":<12} : {v:>6} 条')

#5.1 挑几个关心的输出变量 -----------------
# 方案 B：Real 型 + 只要“本地”或“计算”变量（通常含我们关心的温度、功率）
outs = [v.name for v in info.modelVariables
               if v.type == 'Real' and v.causality in {'output'}]
print('自选变量个数:', len(outs))
print('前 10 个:', outs[:10])
# 快速搜含“load”或“Q”的输入
inputs = [v.name for v in info.modelVariables
          if v.causality == 'input' and 'load' in v.name.lower()]
print('候选输入:', inputs)


# 6. 仿真
try:
    res = simulate_fmu(
        FMU_PATH,
        start_time=START_TIME,
        stop_time=STOP_TIME,
        output_interval=STEP_SIZE,
        output=['Tsupply.T', 'Treturn.T'],
        input=input_signals,
        validate=False
    )

    # 7. 保存结果
    csv_file = 'result.csv'
    write_csv(csv_file, res)
    print('已保存', os.path.abspath(csv_file))

    # 8. 画图
    plt.figure(figsize=(10, 6))
    plt.plot(res['time'], res['Tsupply.T'] - 273.15, label='Supply Temperature [°C]', color='tab:blue')
    plt.plot(res['time'], res['Treturn.T'] - 273.15, label='Return Temperature [°C]', color='tab:red')

    plt.xlabel('Time [s]')
    plt.ylabel('Temperature [°C]')
    plt.title('Cooling System Simulation Results')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fmpy_results.png', dpi=150)
    plt.show()

except Exception as e:
    print(f"仿真过程中发生错误: {e}")
    print("尝试不使用输入信号进行仿真...")

    # 尝试不使用输入信号进行仿真
    res = simulate_fmu(
        FMU_PATH,
        start_time=START_TIME,
        stop_time=STOP_TIME,
        output_interval=STEP_SIZE,
        output=['Tsupply.T', 'Treturn.T'],
        validate=False
    )

    # 保存和绘图代码同上
