"""
Utility Functions for ADC Simulation.

Provides signal generation and helper functions.
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray


def generate_sine(
    n_samples: int = 1024,
    fs: float = 1e6,
    fin: float = 1e3,
    amplitude: float = 0.45,
    offset: float = 0.5,
    coherent: bool = True,
    phase: float = 0.0,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate a sinusoidal test signal.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    fs : float
        Sampling frequency in Hz.
    fin : float
        Input frequency in Hz.
    amplitude : float
        Signal amplitude.
    offset : float
        DC offset.
    coherent : bool
        Adjust frequency for coherent sampling.
    phase : float
        Initial phase in radians.
    
    Returns
    -------
    timestamps : NDArray
        Time points.
    signal : NDArray
        Voltage samples.
    
    Example
    -------
    >>> t, v = generate_sine(n_samples=2048, fs=1e6, fin=10e3)
    >>> result = adc.sim(v)
    """
    timestamps = np.arange(n_samples) / fs
    
    if coherent:
        n_periods = int(np.round(fin * n_samples / fs))
        if n_periods == 0:
            n_periods = 1
        fin_coherent = n_periods * fs / n_samples
    else:
        fin_coherent = fin
    
    signal = offset + amplitude * np.sin(2 * np.pi * fin_coherent * timestamps + phase)
    
    return timestamps, signal


def generate_ramp(
    n_samples: int = 1024,
    vmin: float = 0.0,
    vmax: float = 1.0,
    fs: float = 1e6,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate a ramp test signal.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    vmin : float
        Minimum voltage.
    vmax : float
        Maximum voltage.
    fs : float
        Sampling frequency.
    
    Returns
    -------
    timestamps : NDArray
        Time points.
    signal : NDArray
        Voltage samples.
    """
    timestamps = np.arange(n_samples) / fs
    signal = np.linspace(vmin, vmax, n_samples)
    return timestamps, signal


def generate_multitone(
    n_samples: int = 1024,
    fs: float = 1e6,
    frequencies: Optional[list] = None,
    amplitudes: Optional[list] = None,
    offset: float = 0.5,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Generate a multi-tone test signal.
    
    Parameters
    ----------
    n_samples : int
        Number of samples.
    fs : float
        Sampling frequency.
    frequencies : list
        List of frequencies in Hz.
    amplitudes : list
        List of amplitudes for each tone.
    offset : float
        DC offset.
    
    Returns
    -------
    timestamps : NDArray
        Time points.
    signal : NDArray
        Voltage samples.
    """
    if frequencies is None:
        frequencies = [1e3, 2e3]
    if amplitudes is None:
        amplitudes = [0.2] * len(frequencies)
    
    timestamps = np.arange(n_samples) / fs
    signal = np.full(n_samples, offset)
    
    for freq, amp in zip(frequencies, amplitudes):
        n_periods = int(np.round(freq * n_samples / fs))
        if n_periods == 0:
            n_periods = 1
        freq_coherent = n_periods * fs / n_samples
        signal += amp * np.sin(2 * np.pi * freq_coherent * timestamps)
    
    return timestamps, signal


def ideal_code(
    voltage: float,
    bits: int,
    vref: float = 1.0,
    vmin: float = 0.0,
) -> int:
    """
    Compute ideal digital code for a voltage.
    
    Parameters
    ----------
    voltage : float
        Input voltage.
    bits : int
        ADC resolution.
    vref : float
        Reference voltage.
    vmin : float
        Minimum input voltage.
    
    Returns
    -------
    int
        Ideal digital code.
    """
    lsb = (vref - vmin) / (2 ** bits)
    code = int((voltage - vmin) / lsb)
    return max(0, min(2 ** bits - 1, code))


def codes_to_voltage(
    codes: NDArray[np.int64],
    bits: int,
    vref: float = 1.0,
    vmin: float = 0.0,
) -> NDArray[np.float64]:
    """
    Convert digital codes to reconstructed voltages.
    
    Parameters
    ----------
    codes : NDArray
        Digital codes.
    bits : int
        ADC resolution.
    vref : float
        Reference voltage.
    vmin : float
        Minimum voltage.
    
    Returns
    -------
    NDArray
        Reconstructed voltages.
    """
    lsb = (vref - vmin) / (2 ** bits)
    return vmin + (codes + 0.5) * lsb


def add_noise(
    signal: NDArray[np.float64],
    sigma: float,
) -> NDArray[np.float64]:
    """
    Add Gaussian noise to a signal.
    
    Parameters
    ----------
    signal : NDArray
        Input signal.
    sigma : float
        Noise standard deviation.
    
    Returns
    -------
    NDArray
        Noisy signal.
    """
    return signal + np.random.normal(0, sigma, len(signal))


def thermal_noise_voltage(
    capacitance_fF: float,
    temperature_K: float = 300,
) -> float:
    """
    Compute kT/C thermal noise voltage.
    
    Parameters
    ----------
    capacitance_fF : float
        Capacitance in femtofarads.
    temperature_K : float
        Temperature in Kelvin.
    
    Returns
    -------
    float
        RMS noise voltage in Volts.
    """
    k = 1.38e-23  # Boltzmann constant
    C = capacitance_fF * 1e-15  # Convert to Farads
    return np.sqrt(k * temperature_K / C)


__all__ = [
    "generate_sine",
    "generate_ramp",
    "generate_multitone",
    "ideal_code",
    "codes_to_voltage",
    "add_noise",
    "thermal_noise_voltage",
]
