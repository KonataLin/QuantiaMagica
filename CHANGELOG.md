# Changelog

All notable changes to QuantiaMagica will be documented in this file.

## [1.1.0] - 2026-01-04

### Added

#### 遗传算法优化库 (`quantiamagica.optim`)
- **Gene** - 基因定义，支持多种类型：
  - `float` - 连续浮点数
  - `int` - 整数
  - `choice` - 离散选择
  - `bool` - 布尔值
  - `log_scale` - 对数尺度（用于跨越多个数量级的参数）
- **Individual** - 个体类，支持变异和交叉操作
- **Population** - 种群类，支持锦标赛/轮盘赌选择
- **GeneticOptimizer** - 遗传算法优化器：
  - CUDA/CPU自动检测
  - 多线程并行评估
  - 自适应收敛检测
  - 进度条显示
- **Constraint** - 约束条件支持：
  - `RangeConstraint` - 范围约束
  - `CustomConstraint` - 自定义约束函数

#### 新示例
- `09_genetic_optimizer.py` - 演示如何优化SD ADC系数

### Improved

#### sim_auto 优化
- 改进算法：网格搜索 + 自动参数调整
- 添加CUDA检测提示
- 改进进度显示格式
- 修复SD ADC幅度范围问题，使ENOB更接近理论值
- 优化大样本数性能（搜索/验证分离）

#### SD ADC 参数自适应
- 1-bit量化器：幅度范围0.28-0.39 (根据阶数调整)
- Multi-bit量化器：幅度范围0.3-0.4
- 自动设置n_samples >= OSR*64
- fin限制在信号带宽50%以内

### Fixed
- 移除torch路径硬编码，使用标准import
- 修复HTML文档GPU安装说明样式（白色文字问题）
- 修复SD ADC sim_auto结果反而更差的问题

---

## [1.0.0] - 2026-01-03

### Initial Release
- 事件驱动ADC仿真框架
- SAR ADC、Pipeline ADC、Sigma-Delta ADC支持
- 完整的分析工具（ENOB、SNR、SFDR、THD、INL、DNL）
- JSSC风格绘图
- 8个完整示例
