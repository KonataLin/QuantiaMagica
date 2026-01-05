from quantiamagica.optim import SDCoeffOptimizer

# 创建优化器
optimizer = SDCoeffOptimizer(
    order=2,
    bits=1,
    osr=64,
    fs=1e6
)

# 运行优化
result = optimizer.optimize()

# 打印结果
print(result.summary())
print(f"\n优化前ENOB: {result.baseline_enob:.2f} bits")
print(f"优化后ENOB: {result.best_enob:.2f} bits")
print(f"提升: +{result.improvement:.2f} bits")