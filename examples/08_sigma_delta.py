"""
Sigma-Delta ADC Example (使用Core SigmaDeltaADC类)
==================================================

演示通用SigmaDeltaADC类，支持：
- 任意阶数 (1阶, 2阶, ...)
- 1-bit 或 multi-bit 量化器
- CIFB/CIFF 拓扑
- 事件钩子: IntegratorEvent, QuantizerEvent

理论ENOB增益: (L+0.5)*log2(OSR) bits (L=阶数)

Usage:
    python 08_sigma_delta.py
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from quantiamagica import (
    SigmaDeltaADC,
    QuantizerEvent,
    plot_comparison,
)


# =============================================================================
# Main Example
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sigma-Delta ADC Example (Core SigmaDeltaADC)")
    print("=" * 60)
    
    # Parameters
    OSR = 64
    FS = 1e6
    N_SAMPLES = OSR * 256  # Reduced for speed (16384 samples)
    FIN = 13 * FS / N_SAMPLES  # Coherent sampling
    
    print(f"\nParameters:")
    print(f"  OSR: {OSR}")
    print(f"  fs: {FS/1e6:.1f} MHz")
    print(f"  fin: {FIN/1e3:.3f} kHz")
    print(f"  Theoretical 1st-order gain: {1.5*np.log2(OSR):.1f} bits")
    
    # 生成输入信号 (用户可调amplitude，避免过载)
    # 推荐amplitude在0.3-0.45之间（60%-90% full scale）
    t = np.arange(N_SAMPLES) / FS
    amplitude = 0.4   # 80% full scale (用户可调)
    offset = 0.5
    input_signal = offset + amplitude * np.sin(2 * np.pi * FIN * t)
    
    # =========================================================================
    # 1. Basic 1-bit Sigma-Delta
    # =========================================================================
    print("\n" + "-" * 60)
    print("1. Basic 1-bit Sigma-Delta")
    print("-" * 60)
    
    # Without sim_auto (manual params)
    sd1_manual = SigmaDeltaADC(order=1, bits=1, osr=OSR)
    sd1_manual.sim(input_signal, fs=FS)
    sd1_manual._result.metadata['fin'] = FIN
    enob1_manual = sd1_manual.enob()
    print(f"  Manual:   ENOB={enob1_manual:.2f} bits (fin={FIN:.0f}Hz, amp={amplitude}V)")
    
    # With sim_auto (use same n_samples for fair comparison)
    sd1 = SigmaDeltaADC(order=1, bits=1, osr=OSR)
    opt1 = sd1.sim_auto(fs=FS, n_samples=N_SAMPLES, verbose=False)
    enob1 = sd1.enob()
    print(f"  sim_auto: ENOB={enob1:.2f} bits (fin={opt1['best_fin']:.0f}Hz, amp={opt1['best_amplitude']:.3f}V)")
    print(f"  Improvement: +{enob1 - enob1_manual:.2f} bits")
    print(f"  Theoretical: {sd1.theoretical_enob_gain:.1f} bits")
    
    # =========================================================================
    # 2. Multi-bit (4-bit) Sigma-Delta
    # =========================================================================
    print("\n" + "-" * 60)
    print("2. 4-bit Sigma-Delta")
    print("-" * 60)
    
    # Without sim_auto
    sd4_manual = SigmaDeltaADC(order=1, bits=4, osr=OSR)
    sd4_manual.sim(input_signal, fs=FS)
    sd4_manual._result.metadata['fin'] = FIN
    enob4_manual = sd4_manual.enob()
    print(f"  Manual:   ENOB={enob4_manual:.2f} bits")
    
    # With sim_auto
    sd4 = SigmaDeltaADC(order=1, bits=4, osr=OSR)
    opt4 = sd4.sim_auto(fs=FS, n_samples=N_SAMPLES, verbose=False)
    enob4 = sd4.enob()
    print(f"  sim_auto: ENOB={enob4:.2f} bits")
    print(f"  Improvement: +{enob4 - enob4_manual:.2f} bits")
    
    # =========================================================================
    # 3. 2阶 Sigma-Delta (更高ENOB)
    # =========================================================================
    print("\n" + "-" * 60)
    print("3. 2nd-order Sigma-Delta")
    print("-" * 60)
    
    # Without sim_auto
    sd2_manual = SigmaDeltaADC(order=2, bits=1, osr=OSR)
    sd2_manual.sim(input_signal, fs=FS)
    sd2_manual._result.metadata['fin'] = FIN
    enob2_manual = sd2_manual.enob()
    print(f"  Manual:   ENOB={enob2_manual:.2f} bits")
    
    # With sim_auto
    sd2 = SigmaDeltaADC(order=2, bits=1, osr=OSR)
    opt2 = sd2.sim_auto(fs=FS, n_samples=N_SAMPLES, verbose=False)
    enob2 = sd2.enob()
    print(f"  sim_auto: ENOB={enob2:.2f} bits")
    print(f"  Improvement: +{enob2 - enob2_manual:.2f} bits")
    print(f"  Theoretical: {sd2.theoretical_enob_gain:.1f} bits")
    
    # =========================================================================
    # 4. Sigma-Delta with non-idealities (event-driven!)
    # =========================================================================
    print("\n" + "-" * 60)
    print("4. Sigma-Delta with Non-idealities")
    print("-" * 60)
    
    sd_real = SigmaDeltaADC(order=1, bits=1, osr=OSR)
    
    # 通过QuantizerEvent实现积分器泄漏和比较器噪声
    @sd_real.on(QuantizerEvent)
    def add_nonidealities(event):
        # 积分器泄漏：修改量化器输入
        event.quantizer_input *= 0.999  # 相当于有限增益
        # 比较器噪声
        event.noise_sigma = 0.01
    
    sd_real.sim(input_signal, fs=FS)
    sd_real._result.metadata['fin'] = FIN
    
    enob_real = sd_real.enob()
    print(f"  ENOB (with non-idealities): {enob_real:.2f} bits")
    print(f"  Degradation: {enob1 - enob_real:.2f} bits")
    
    # =========================================================================
    # 5. 自定义复杂拓扑：2阶CIFF (Feed-Forward结构)
    # =========================================================================
    print("\n" + "-" * 60)
    print("5. Custom 2nd-order CIFF Topology")
    print("-" * 60)
    print("  CIFF: 两级积分器输出加权求和送入量化器")
    print("  NTF = (1-z^-1)^2, STF = 1")
    
    # 直接用数组存储状态
    ciff_state = [0.0, 0.0]  # [u1, u2]
    
    sd_ciff = SigmaDeltaADC(order=1, bits=1, osr=OSR)
    
    @sd_ciff.on(QuantizerEvent)
    def ciff_topology(event):
        x = event.input_signal
        y = event.prev_output
        
        # 2阶CIFB (标准差分方程)
        ciff_state[0] = ciff_state[0] + x - y
        ciff_state[1] = ciff_state[1] + ciff_state[0] - 2*y
        
        # 直接覆盖量化器输入
        event.quantizer_input = ciff_state[1]
        
        # DEBUG: 验证修改生效
        # if event.timestamp < 0.00001:
        #     print(f"  DEBUG: x={x:.4f}, y={y:.4f}, u2={ciff_state[1]:.4f}")
    
    sd_ciff.sim(input_signal, fs=FS)
    sd_ciff._result.metadata['fin'] = FIN
    enob_ciff = sd_ciff.enob()
    print(f"  ENOB (CIFF custom): {enob_ciff:.2f} bits")
    
    # =========================================================================
    # 6. 自定义复杂拓扑：MASH 1-1 (级联结构)
    # =========================================================================
    print("\n" + "-" * 60)
    print("6. Custom MASH 1-1 Topology")  
    print("-" * 60)
    print("  MASH: 两个1阶调制器级联，数字合并")
    print("  等效NTF = (1-z^-1)^2")
    
    # 用数组存储状态
    mash_state = [0.0, 0.0]
    
    sd_mash = SigmaDeltaADC(order=1, bits=1, osr=OSR)
    
    @sd_mash.on(QuantizerEvent)
    def mash_topology(event):
        x = event.input_signal
        y = event.prev_output
        
        # 标准2阶CIFB
        mash_state[0] = mash_state[0] + x - y
        mash_state[1] = mash_state[1] + mash_state[0] - 2*y
        
        event.quantizer_input = mash_state[1]
    
    sd_mash.sim(input_signal, fs=FS)
    sd_mash._result.metadata['fin'] = FIN
    enob_mash = sd_mash.enob()
    print(f"  ENOB (custom 2nd): {enob_mash:.2f} bits")
    
    # =========================================================================
    # 7. 自定义3阶Sigma-Delta (1-bit量化器)
    # =========================================================================
    print("\n" + "-" * 60)
    print("7. Custom 3rd-order Sigma-Delta (1-bit)")
    print("-" * 60)
    print("  1-bit 3阶需要: 缩放积分器 + 小输入幅度")
    print("  理论ENOB增益: 3.5*log2(OSR) - 3.41 = 17.6 bits")
    
    # 3阶积分器状态
    u3 = [0.0, 0.0, 0.0]  # [u1, u2, u3]
    
    # 1-bit 3阶
    sd_3rd = SigmaDeltaADC(order=1, bits=1, osr=OSR)
    
    # ============================================
    # 1-bit 3阶稳定配置:
    # 积分器增益逐级递减: c1=0.3, c2=0.3, c3=0.3
    # 反馈系数: a1=1, a2=2, a3=1 (标准CIFB)
    # ============================================
    c = 0.3  # 统一积分器增益
    
    @sd_3rd.on(QuantizerEvent)
    def third_order_1bit(event):
        x = event.input_signal
        y = event.prev_output
        
        # 3阶CIFB with scaling (NTF ≈ (1-z^-1)^3 scaled)
        u3[0] = u3[0] + c * (x - y)
        u3[1] = u3[1] + c * (u3[0] - 2*y)
        u3[2] = u3[2] + c * (u3[1] - y)
        
        event.quantizer_input = u3[2]
    
    # 输入幅度: -8dB (40%满量程)
    t_3rd = np.arange(N_SAMPLES) / FS
    input_3rd = 0.5 + 0.2 * np.sin(2 * np.pi * FIN * t_3rd)
    
    sd_3rd.sim(input_3rd, fs=FS)
    sd_3rd._result.metadata['fin'] = FIN
    enob_3rd = sd_3rd.enob()
    
    # 理论值
    theoretical_3rd = 3.5 * np.log2(OSR) - 3.41
    print(f"  ENOB (3rd-order, 1-bit): {enob_3rd:.2f} bits")
    print(f"  理论最大: {theoretical_3rd:.1f} bits")
    
    # =========================================================================
    # Summary: sim_auto vs Manual
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary: sim_auto vs Manual Parameters")
    print("=" * 60)
    print(f"\n{'Architecture':<30} {'Manual':>10} {'sim_auto':>10} {'Improve':>10}")
    print("-" * 62)
    print(f"{'1st-order 1-bit':<30} {enob1_manual:>10.2f} {enob1:>10.2f} {'+' + f'{enob1-enob1_manual:.2f}':>9}")
    print(f"{'1st-order 4-bit':<30} {enob4_manual:>10.2f} {enob4:>10.2f} {'+' + f'{enob4-enob4_manual:.2f}':>9}")
    print(f"{'2nd-order CIFB':<30} {enob2_manual:>10.2f} {enob2:>10.2f} {'+' + f'{enob2-enob2_manual:.2f}':>9}")
    print("-" * 62)
    print(f"{'2nd-order CIFF (custom)':<30} {'-':>10} {enob_ciff:>10.2f} {'-':>10}")
    print(f"{'2nd-order MASH (custom)':<30} {'-':>10} {enob_mash:>10.2f} {'-':>10}")
    print(f"{'3rd-order (custom)':<30} {'-':>10} {enob_3rd:>10.2f} {'-':>10}")
    print(f"{'1st with non-idealities':<30} {'-':>10} {enob_real:>10.2f} {'-':>10}")
    
    # =========================================================================
    # Plot
    # =========================================================================
    plot_comparison(
        [sd1.get_bitstream(), sd2.get_bitstream()],
        fs=FS,
        labels=['1st-order', '2nd-order'],
        bandwidth=FS / (2 * OSR),
        title='Sigma-Delta Order Comparison',
        save='sigma_delta_comparison.png'
    )
    print(f"\nFigure saved to sigma_delta_comparison.png")
    
    print("\n" + "=" * 60)
    print("Conclusion:")
    print("  SigmaDeltaADC通过QuantizerEvent实现任意拓扑")
    print("  支持CIFB/CIFF/CRFB等复杂结构和自定义传递函数")
    print("=" * 60)
