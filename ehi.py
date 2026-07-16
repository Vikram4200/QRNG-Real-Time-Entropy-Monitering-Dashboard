import math
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


st.set_page_config(
    page_title="QRNG Entropy Health Dashboard",
    page_icon="QRNG",
    layout="wide",
)


def generate_qrng_bits(shots):
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.measure_all()

    simulator = AerSimulator()
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    bit_sequence = []
    for outcome, count in counts.items():
        bit_sequence.extend([outcome] * count)

    np.random.shuffle(bit_sequence)
    return "".join(bit_sequence), counts


def clamp(value, low=0, high=100):
    return max(low, min(high, value))


def shannon_entropy(bits):
    counts = Counter(bits)
    total = len(bits)
    probs = [count / total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def min_entropy(bits):
    counts = Counter(bits)
    total = len(bits)
    probs = [count / total for count in counts.values()]
    return -math.log2(max(probs))


def collision_entropy(bits):
    counts = Counter(bits)
    total = len(bits)
    probs = [count / total for count in counts.values()]
    return -math.log2(sum(p * p for p in probs))


def frequency_test(bits):
    n = len(bits)
    s = sum(1 if bit == "1" else -1 for bit in bits)
    sobs = abs(s) / math.sqrt(n)
    return math.erfc(sobs / math.sqrt(2))


def runs_test(bits):
    n = len(bits)
    pi = bits.count("1") / n
    tau = 2 / math.sqrt(n)

    if abs(pi - 0.5) >= tau:
        return 0.0

    runs = 1
    for i in range(1, n):
        if bits[i] != bits[i - 1]:
            runs += 1

    numerator = abs(runs - 2 * n * pi * (1 - pi))
    denominator = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    return math.erfc(numerator / denominator)


def autocorrelation(bits, lag=1):
    data = np.array([int(bit) for bit in bits])

    if len(data) <= lag:
        return 0.0

    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
    if np.isnan(corr):
        return 0.0

    return corr


def score_high_good(value, worst, excellent):
    return clamp(((value - worst) / (excellent - worst)) * 100)


def score_low_good(value, excellent, worst):
    return clamp(((worst - value) / (worst - excellent)) * 100)


def score_p_value(p_value):
    return clamp(((p_value - 0.01) / (0.10 - 0.01)) * 100)


def status_from_ehi(ehi):
    if ehi >= 90:
        return "Excellent"
    if ehi >= 75:
        return "Good"
    if ehi >= 60:
        return "Average"
    if ehi >= 40:
        return "Warning"
    return "Worst"


def calculate_ehi(bits, previous_bias=None):
    shannon = shannon_entropy(bits)
    min_ent = min_entropy(bits)
    collision = collision_entropy(bits)

    frequency_p = frequency_test(bits)
    runs_p = runs_test(bits)
    autocorr = abs(autocorrelation(bits, lag=1))

    p1 = bits.count("1") / len(bits)
    bias = abs(p1 - 0.5)
    noise = autocorr
    drift = 0.0 if previous_bias is None else abs(bias - previous_bias)

    shannon_score = score_high_good(shannon, 0.90, 0.99)
    min_score = score_high_good(min_ent, 0.85, 0.97)
    collision_score = score_high_good(collision, 0.88, 0.98)
    frequency_score = score_p_value(frequency_p)
    runs_score = score_p_value(runs_p)
    autocorr_score = score_low_good(autocorr, 0.01, 0.10)
    bias_score = score_low_good(bias, 0.005, 0.05)
    noise_score = score_low_good(noise, 0.005, 0.05)
    drift_score = score_low_good(drift, 0.005, 0.05)

    ehi = (
        0.12 * shannon_score
        + 0.20 * min_score
        + 0.13 * collision_score
        + 0.10 * frequency_score
        + 0.10 * runs_score
        + 0.10 * autocorr_score
        + 0.10 * bias_score
        + 0.07 * noise_score
        + 0.08 * drift_score
    )

    return {
        "shannon": shannon,
        "min_entropy": min_ent,
        "collision": collision,
        "frequency_p": frequency_p,
        "runs_p": runs_p,
        "autocorrelation": autocorr,
        "bias": bias,
        "noise": noise,
        "drift": drift,
        "ehi": ehi,
        "status": status_from_ehi(ehi),
    }, bias


def status_color(status):
    colors = {
        "Excellent": "#1d561b",
        "Good": "#7df392",
        "Average": "#f08c00",
        "Warning": "#e03131",
        "Worst": "#a61e4d",
    }
    return colors.get(status, "#333333")


def draw_live_graph(history):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["time"], history["ehi"], marker="o", linewidth=2, color="#16803c")
    ax.axhline(90, color="#1864ab", linestyle="--", label="Excellent")
    ax.axhline(75, color="#2f9e44", linestyle="--", label="Good")
    ax.axhline(60, color="#f08c00", linestyle="--", label="Average")
    ax.axhline(40, color="#e03131", linestyle="--", label="Warning")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("EHI (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return fig


st.title("QRNG Real-Time Entropy Health Dashboard")

with st.sidebar:
    st.header("Dashboard Settings")
    sample_size = st.number_input("Bits generation shots", min_value=100, max_value=100000, value=1000, step=100)
    delay = st.slider("Update delay (seconds)", min_value=1, max_value=10, value=5)
    iterations = st.slider("Number of updates", min_value=1, max_value=100, value=20)
    start = st.button("Start Live Dashboard", type="primary")
    single_run = st.button("Generate One Reading")

if "ehi_history" not in st.session_state:
    st.session_state.ehi_history = []
if "time_history" not in st.session_state:
    st.session_state.time_history = []
if "previous_bias" not in st.session_state:
    st.session_state.previous_bias = None

health_box = st.empty()
panel_area = st.empty()
graph_area = st.empty()


def update_dashboard(step_number):
    bits, counts = generate_qrng_bits(sample_size)
    results, st.session_state.previous_bias = calculate_ehi(bits, st.session_state.previous_bias)

    elapsed_time = step_number * delay
    st.session_state.time_history.append(elapsed_time)
    st.session_state.ehi_history.append(results["ehi"])

    color = status_color(results["status"])

    with health_box.container():
        st.markdown(
            f"""
            <div style="padding:22px;border-radius:8px;background:#f8f9fa;border:1px solid #dee2e6">
                <h2 style="margin:0;color:{color};">EHI = {results["ehi"]:.2f}%</h2>
                <h3 style="margin:6px 0 0 0;">Status = {results["status"]}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    top_col1, top_col2 = st.columns([1, 2])

    with top_col1:
        st.metric("Entropy Health Index", f"{results['ehi']:.2f}%")

    with top_col2:
        if results["status"] == "Excellent":
            st.success(f"Status: {results['status']}")
        elif results["status"] == "Good":
            st.success(f"Status: {results['status']}")
        elif results["status"] == "Average":
            st.warning(f"Status: {results['status']}")
        else:
            st.error(f"Status: {results['status']}")

    with panel_area.container():
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.subheader("Entropy Values")
            st.metric("Shannon Entropy", f"{results['shannon']:.4f}")
            st.metric("Min Entropy", f"{results['min_entropy']:.4f}")
            st.metric("Collision Entropy", f"{results['collision']:.4f}")

        with col2:
            st.subheader("Statistical Tests")
            st.metric("Frequency p-value", f"{results['frequency_p']:.4f}")
            st.metric("Runs p-value", f"{results['runs_p']:.4f}")
            st.metric("Autocorrelation", f"{results['autocorrelation']:.4f}")

        with col3:
            st.subheader("Degradation")
            st.metric("Bias", f"{results['bias'] * 100:.2f}%")
            st.metric("Noise", f"{results['noise'] * 100:.2f}%")
            st.metric("Drift", f"{results['drift'] * 100:.2f}%")

        with col4:
            st.subheader("2-Qubit Counts")
            st.dataframe(pd.DataFrame(counts.items(), columns=["Outcome", "Count"]), hide_index=True)

    with graph_area.container():
        history = pd.DataFrame(
            {
                "time": st.session_state.time_history,
                "ehi": st.session_state.ehi_history,
            }
        )
        st.subheader("Live EHI Graph")
        st.pyplot(draw_live_graph(history))


if single_run:
    update_dashboard(len(st.session_state.ehi_history))

if start:
    for step in range(iterations):
        update_dashboard(step)
        time.sleep(delay)

if not start and not single_run:
    st.info("Sidebar se 'Generate One Reading' ya 'Start Live Dashboard' click karo.")
