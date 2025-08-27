import matplotlib.pyplot as plt
import numpy as np

# ====================== 1. 定义基础参数 ======================
BIT_WIDTH = 33_554_432  # 全连接层总数据量（bit）：1024*1024*32
NUM_PE = 16             # PE 数量
CLOCK_FREQ = 1e9       # 时钟频率（1 GHz，用于估算时间）

# 能量模型（pJ）
ENERGY_DRAM_READ = 120    # DRAM 读 1 bit 能量
ENERGY_SRAM_READ = 1.3    # SRAM 读 1 bit 能量
ENERGY_PE_OP = 0.3        # PE 单次乘加操作能量
ENERGY_IO_TRANSFER = 0.1  # IO 传输 1 bit 能量

# 模块功率转换（pJ → W：1 W = 1 J/s = 1e12 pJ/s）
def energy_to_power(energy_pj, time_s):
    return (energy_pj * 1e-12) / time_s if time_s > 0 else 0


# ====================== 2. 阶段 1：DRAM → SRAM（数据加载） ======================
def stage_dram_to_sram():
    # 1. DRAM 读操作能量
    energy_dram = BIT_WIDTH * ENERGY_DRAM_READ  # pJ
    
    # 2. IO 传输能量（DRAM → Chiplet 需经过 IO）
    energy_io = BIT_WIDTH * ENERGY_IO_TRANSFER  # pJ
    
    # 3. 片上 SRAM 写操作（简化为读能量等同）
    energy_sram_write = BIT_WIDTH * ENERGY_SRAM_READ  # pJ
    
    # 总能量
    total_energy = energy_dram + energy_io + energy_sram_write  # pJ
    
    # 时间估算（假设带宽限制：10 GB/s = 8e10 bit/s）
    bandwidth = 8e10  # bit/s
    time = BIT_WIDTH / bandwidth  # s
    
    # 功率
    power = energy_to_power(total_energy, time)  # W
    
    return {
        "stage": "DRAM → SRAM",
        "energy_pj": total_energy,
        "time_s": time,
        "power_w": power
    }


# ====================== 3. 阶段 2：PE 阵列计算 ======================
def stage_pe_compute():
    # 全连接层计算量：1024×1024 次乘加（简化为每个权值 1 次 OP）
    num_ops = 1024 * 1024  # 总操作数
    
    # 16 个 PE 并行，每个 PE 承担的操作数
    ops_per_pe = num_ops / NUM_PE
    
    # 每个 PE 的能量消耗
    energy_per_pe = ops_per_pe * ENERGY_PE_OP  # pJ/PE
    total_energy = energy_per_pe * NUM_PE       # 总能量（pJ）
    
    # 时间估算（1 个 PE 耗时：ops_per_pe / CLOCK_FREQ）
    time = (ops_per_pe / CLOCK_FREQ)  # s（假设 1 周期 1 操作）
    
    # 功率
    power = energy_to_power(total_energy, time)  # W
    
    return {
        "stage": "PE Compute",
        "energy_pj": total_energy,
        "time_s": time,
        "power_w": power
    }


# ====================== 4. 阶段 3：结果回写（SRAM） ======================
def stage_write_back():
    # 假设结果数据量与输入相同（33,554,432 bit）
    energy_sram_write = BIT_WIDTH * ENERGY_SRAM_READ  # pJ（简化为写能量等同读）
    time = BIT_WIDTH / (8e10)  # 假设写带宽同读（10 GB/s）
    
    power = energy_to_power(energy_sram_write, time)  # W
    
    return {
        "stage": "Write Back",
        "energy_pj": energy_sram_write,
        "time_s": time,
        "power_w": power
    }


# ====================== 5. 模拟功耗迹线 ======================
if __name__ == "__main__":
    # 执行各阶段
    stage1 = stage_dram_to_sram()
    stage2 = stage_pe_compute()
    stage3 = stage_write_back()

    # 整理结果
    stages = [stage1, stage2, stage3]
    stage_names = [s["stage"] for s in stages]
    stage_powers = [s["power_w"] for s in stages]
    stage_times = [s["time_s"] for s in stages]

    # 打印各阶段 summary
    for s in stages:
        print(f"[{s['stage']}] 功耗: {s['power_w']:.2f} W, 耗时: {s['time_s']:.6f} s, 能量: {s['energy_pj']/1e6:.2f} mJ")

    # 绘制功耗迹线（时间轴拼接）
    time_points = np.cumsum([0] + stage_times)
    power_trace = np.repeat(stage_powers, 2)  # 简化为阶跃曲线
    time_trace = np.concatenate([[time_points[i], time_points[i+1]] for i in range(len(time_points)-1)])

    plt.figure(figsize=(10, 5))
    plt.step(time_trace, power_trace, where='post', color='b', linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("Power Trace: 1024×1024 全连接层任务")
    plt.xticks(time_points)
    plt.grid(True)
    plt.show()