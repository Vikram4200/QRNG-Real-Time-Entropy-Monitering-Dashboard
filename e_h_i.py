import math
import time
from collections import Counter
import hashlib
import numpy as np
import pandas as pd
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ============================================================
# QRNG ENTROPY HEALTH INDEX DASHBOARD
# Step 5: Threshold Design + Real-Time Dashboard
# ============================================================


st.set_page_config(
    page_title="QRNG Entropy Health Index Dashboard",
    page_icon="Q",
    layout="wide",
)


# ============================================================
# 1. QRNG BIT GENERATION USING QISKIT
# ============================================================


@st.cache_resource
def get_simulator():
    return AerSimulator()


def generate_qrng_bits(shots=1000):
    """
    2-qubit QRNG circuit.
    Each shot gives 2 bits: 00, 01, 10, or 11.
    """
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    qc.measure_all()

    simulator = get_simulator()
    compiled = transpile(qc, simulator)
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    bit_blocks = []
    for outcome, count in counts.items():
        bit_blocks.extend([outcome] * count)

    np.random.shuffle(bit_blocks)
    bitstream = "".join(bit_blocks)

    return bitstream, counts


# ============================================================
# 2. ENTROPY CALCULATIONS
# ============================================================
def generate_prng_bits(length):
    """
    Classical pseudo-random bit generator for QRNG vs PRNG comparison.
    """
    return "".join(str(bit) for bit in np.random.randint(0, 2, length))


def apply_bias(bits, probability_of_one):
    """
    Bias model: force the stream to follow a selected P(1).
    """
    return "".join("1" if np.random.random() < probability_of_one else "0" for _ in bits)


def apply_noise(bits, noise_level):
    """
    Noise model: randomly flip bits.
    """
    noisy_bits = []
    for bit in bits:
        if np.random.random() < noise_level:
            noisy_bits.append("0" if bit == "1" else "1")
        else:
            noisy_bits.append(bit)
    return "".join(noisy_bits)


def apply_drift(bits, drift_strength):
    """
    Drift model: P(1) slowly increases across the bitstream.
    """
    if drift_strength <= 0:
        return bits

    drifted_bits = []
    length = len(bits)
    for index, bit in enumerate(bits):
        local_bias = 0.5 + drift_strength * (index / max(length - 1, 1))
        if np.random.random() < local_bias:
            drifted_bits.append("1")
        else:
            drifted_bits.append("0")
    return "".join(drifted_bits)


def von_neumann_extractor(bits):
    """
    Von Neumann Extractor:
    01 -> 0, 10 -> 1, 00 and 11 are discarded.
    """
    extracted = []
    for i in range(0, len(bits) - 1, 2):
        pair = bits[i : i + 2]
        if pair == "01":
            extracted.append("0")
        elif pair == "10":
            extracted.append("1")
    return "".join(extracted)


def hash_extractor(bits, output_length=None):
    """
    Hash extractor using SHA-256.
    It compresses bit chunks into hash output bits.
    """
    if output_length is None:
        output_length = len(bits)

    output_bits = []
    counter = 0

    while len(output_bits) < output_length:
        block = bits[counter * 256 : (counter + 1) * 256]
        if not block:
            block = bits + str(counter)

        block_int = int(block, 2) if set(block).issubset({"0", "1"}) else 0
        block_bytes = block_int.to_bytes(max(1, (len(block) + 7) // 8), byteorder="big")
        digest = hashlib.sha256(block_bytes + counter.to_bytes(4, byteorder="big")).digest()

        for byte in digest:
            output_bits.extend(format(byte, "08b"))

        counter += 1

    return "".join(output_bits[:output_length])



def shannon_entropy(bits):
    if len(bits) == 0:
        return 0.0
    counts = Counter(bits)
    total = len(bits)
    probs = [count / total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs)


def min_entropy(bits):
    if len(bits) == 0:
        return 0.0
    counts = Counter(bits)
    total = len(bits)
    probs = [count / total for count in counts.values()]
    return -math.log2(max(probs))


def collision_entropy(bits):
    if len(bits) == 0:
        return 0.0
    counts = Counter(bits)
    total = len(bits)
    probs = [count / total for count in counts.values()]
    return -math.log2(sum(p * p for p in probs))


# ============================================================
# 3. STATISTICAL TESTS
# ============================================================


def frequency_test(bits):
    """
    Frequency Monobit Test.
    H0: number of 0s and 1s should be approximately equal.
    p-value >= 0.01 means pass.
    """
    n = len(bits)
    if n == 0:
        return 0.0
    s = sum(1 if bit == "1" else -1 for bit in bits)
    s_obs = abs(s) / math.sqrt(n)
    p_value = math.erfc(s_obs / math.sqrt(2))
    return p_value


def runs_test(bits):
    """
    Runs Test.
    H0: sequence oscillates between 0 and 1 naturally.
    p-value >= 0.01 means pass.
    """
    n = len(bits)
    if n == 0:
        return 0.0
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
    p_value = math.erfc(numerator / denominator)

    return p_value


def autocorrelation(bits, lag=1):
    """
    Autocorrelation should be close to zero.
    """
    data = np.array([int(bit) for bit in bits])

    if len(data) <= lag:
        return 0.0

    corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]

    if np.isnan(corr):
        return 0.0

    return corr


# ============================================================
# 4. THRESHOLD CATEGORY DESIGN
# ============================================================


def category_high_good(value, thresholds):
    """
    Higher value is better.
    Used for entropy.
    """
    if value >= thresholds["excellent"]:
        return "Excellent"
    if value >= thresholds["good"]:
        return "Good"
    if value >= thresholds["average"]:
        return "Average"
    if value >= thresholds["warning"]:
        return "Warning"
    return "Worst"


def category_low_good(value, thresholds):
    """
    Lower value is better.
    Used for autocorrelation, bias, noise, drift.
    """
    if value <= thresholds["excellent"]:
        return "Excellent"
    if value <= thresholds["good"]:
        return "Good"
    if value <= thresholds["average"]:
        return "Average"
    if value <= thresholds["warning"]:
        return "Warning"
    return "Worst"


def category_p_value(p_value):
    """
    p-value category for Frequency Test and Runs Test.
    NIST commonly uses p-value >= 0.01 as pass.
    """
    if p_value >= 0.10:
        return "Excellent"
    if p_value >= 0.05:
        return "Good"
    if p_value >= 0.03:
        return "Average"
    if p_value >= 0.01:
        return "Warning"
    return "Worst"


THRESHOLDS = {
    "shannon": {"warning": 0.900, "average": 0.940, "good": 0.970, "excellent": 0.990},
    "min_entropy": {"warning": 0.850, "average": 0.900, "good": 0.940, "excellent": 0.970},
    "collision": {"warning": 0.880, "average": 0.930, "good": 0.960, "excellent": 0.980},
    "autocorrelation": {"excellent": 0.010, "good": 0.020, "average": 0.050, "warning": 0.100},
    "bias": {"excellent": 0.005, "good": 0.010, "average": 0.020, "warning": 0.050},
    "noise": {"excellent": 0.005, "good": 0.010, "average": 0.020, "warning": 0.050},
    "drift": {"excellent": 0.005, "good": 0.010, "average": 0.020, "warning": 0.050},
}


# ============================================================
# 5. EHI SCORING FORMULA
# ============================================================


def clamp(value, low=0.0, high=100.0):
    return max(low, min(high, value))


def score_high_good(value, worst, excellent):
    score = ((value - worst) / (excellent - worst)) * 100
    return clamp(score)


def score_low_good(value, excellent, worst):
    score = ((worst - value) / (worst - excellent)) * 100
    return clamp(score)


def score_p_value(p_value):
    score = ((p_value - 0.01) / (0.10 - 0.01)) * 100
    return clamp(score)


def final_status(ehi):
    if ehi >= 90:
        return "Excellent"
    if ehi >= 75:
        return "Good"
    if ehi >= 60:
        return "Average"
    if ehi >= 40:
        return "Warning"
    return "Worst"


def status_color(status):
    colors = {
        "Excellent": "#16803c",
        "Good": "#2f9e44",
        "Average": "#f08c00",
        "Warning": "#e03131",
        "Worst": "#a61e4d",
    }
    return colors[status]


def pass_fail_p_value(p_value):
    return "Pass" if p_value >= 0.01 else "Fail"


def pass_fail_autocorr(value):
    return "Pass" if abs(value) <= 0.05 else "Fail"


def calculate_all_metrics(bits, previous_bias=None):
    shannon = shannon_entropy(bits)
    min_ent = min_entropy(bits)
    collision = collision_entropy(bits)

    frequency_p = frequency_test(bits)
    runs_p = runs_test(bits)
    autocorr = abs(autocorrelation(bits, lag=1))

    p1 = bits.count("1") / len(bits)
    bias = abs(p1 - 0.5)

    # In this simulated dashboard, noise is estimated using autocorrelation.
    # In real hardware, this can be replaced by measured detector/electronic noise.
    noise = autocorr

    # Drift is measured as change in bias between current and previous window.
    if previous_bias is None:
        drift = 0.0
    else:
        drift = abs(bias - previous_bias)

    shannon_score = score_high_good(shannon, 0.900, 0.990)
    min_score = score_high_good(min_ent, 0.850, 0.970)
    collision_score = score_high_good(collision, 0.880, 0.980)

    frequency_score = score_p_value(frequency_p)
    runs_score = score_p_value(runs_p)
    autocorr_score = score_low_good(autocorr, 0.010, 0.100)

    bias_score = score_low_good(bias, 0.005, 0.050)
    noise_score = score_low_good(noise, 0.005, 0.050)
    drift_score = score_low_good(drift, 0.005, 0.050)

    # Weightage:
    # Entropy section = 45%
    # Statistical test section = 30%
    # Degradation section = 25%
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

    status = final_status(ehi)

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
        "status": status,
        "shannon_category": category_high_good(shannon, THRESHOLDS["shannon"]),
        "min_category": category_high_good(min_ent, THRESHOLDS["min_entropy"]),
        "collision_category": category_high_good(collision, THRESHOLDS["collision"]),
        "frequency_category": category_p_value(frequency_p),
        "runs_category": category_p_value(runs_p),
        "autocorr_category": category_low_good(autocorr, THRESHOLDS["autocorrelation"]),
        "bias_category": category_low_good(bias, THRESHOLDS["bias"]),
        "noise_category": category_low_good(noise, THRESHOLDS["noise"]),
        "drift_category": category_low_good(drift, THRESHOLDS["drift"]),
    }, bias


def compact_metric_row(name, bits):
    metrics, _ = calculate_all_metrics(bits, None)
    return {
        "Source": name,
        "Length": len(bits),
        "Shannon": round(metrics["shannon"], 4),
        "Min Entropy": round(metrics["min_entropy"], 4),
        "Collision": round(metrics["collision"], 4),
        "Frequency": pass_fail_p_value(metrics["frequency_p"]),
        "Runs": pass_fail_p_value(metrics["runs_p"]),
        "Autocorrelation": round(metrics["autocorrelation"], 4),
        "Bias (%)": round(metrics["bias"] * 100, 2),
        "EHI (%)": round(metrics["ehi"], 2),
        "Status": metrics["status"],
    }


def build_comparison_table(qrng_bits):
    prng_bits = generate_prng_bits(len(qrng_bits))
    von_neumann_bits = von_neumann_extractor(qrng_bits)
    hash_bits = hash_extractor(qrng_bits, output_length=len(qrng_bits))

    rows = [compact_metric_row("QRNG Raw", qrng_bits)]

    if len(von_neumann_bits) > 0:
        rows.append(compact_metric_row("QRNG + Von Neumann Extractor", von_neumann_bits))
    else:
        rows.append(
            {
                "Source": "QRNG + Von Neumann Extractor",
                "Length": 0,
                "Shannon": 0.0,
                "Min Entropy": 0.0,
                "Collision": 0.0,
                "Frequency": "Fail",
                "Runs": "Fail",
                "Autocorrelation": 0.0,
                "Bias (%)": 0.0,
                "EHI (%)": 0.0,
                "Status": "Worst",
            }
        )

    rows.append(compact_metric_row("QRNG + Hash Extractor", hash_bits))
    rows.append(compact_metric_row("PRNG", prng_bits))

    return pd.DataFrame(rows)

# ============================================================
# 6. STREAMLIT DASHBOARD UI
# ============================================================


st.title("Quantum Random Number Generator - Entropy Health Index")

with st.sidebar:
    st.header("Real-Time Settings")
    shots = st.number_input("Generate shots every cycle", min_value=100, max_value=100000, value=1000, step=100)
    delay = st.slider("Update interval in seconds", min_value=1, max_value=10, value=5)
    updates = st.slider("Number of updates", min_value=1, max_value=100, value=20)

    st.divider()
    st.header("Degradation Model")
    enable_degradation = st.checkbox("Apply Bias / Noise / Drift", value=False)
    bias_probability = st.slider("Bias: Probability of 1", min_value=0.50, max_value=0.95, value=0.50, step=0.01)
    noise_level = st.slider("Noise: Bit flip probability", min_value=0.00, max_value=0.50, value=0.00, step=0.01)
    drift_strength = st.slider("Drift strength over time", min_value=0.00, max_value=0.40, value=0.00, step=0.01)

    st.divider()
    one_reading = st.button("Generate One Reading")
    start_live = st.button("Start Real-Time Monitoring", type="primary")
    reset_history = st.button("Reset History")


if "previous_bias" not in st.session_state:
    st.session_state.previous_bias = None
if "ehi_history" not in st.session_state:
    st.session_state.ehi_history = []
if "time_history" not in st.session_state:
    st.session_state.time_history = []
if "cycle_count" not in st.session_state:
    st.session_state.cycle_count = 0

if reset_history:
    st.session_state.previous_bias = None
    st.session_state.ehi_history = []
    st.session_state.time_history = []
    st.session_state.cycle_count = 0
    st.success("History reset successfully.")


health_panel = st.empty()
dashboard_panel = st.empty()
graph_panel = st.empty()
threshold_panel = st.empty()


def render_dashboard(metrics, counts, bits):
    status = metrics["status"]
    color = status_color(status)

    with health_panel.container():
        st.markdown(
            f"""
            <div style="padding:22px;border-radius:8px;background:#f8f9fa;border:1px solid #dee2e6;">
                <h1 style="margin:0;color:{color};">EHI = {metrics["ehi"]:.2f}%</h1>
                <h2 style="margin:6px 0 0 0;color:{color};">Status = {status}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with dashboard_panel.container():
        panel1, panel2, panel3, panel4 = st.columns(4)

        with panel1:
            st.subheader("Panel 1: Current Health")
            st.metric("EHI", f"{metrics['ehi']:.2f}%")
            if status in ["Excellent", "Good"]:
                st.success(f"Status: {status}")
            elif status == "Average":
                st.warning(f"Status: {status}")
            else:
                st.error(f"Status: {status}")

        with panel2:
            st.subheader("Panel 2: Entropy Values")
            st.metric("Shannon Entropy", f"{metrics['shannon']:.4f}")
            st.write(f"Status: **{metrics['shannon_category']}**")

            st.metric("Min Entropy", f"{metrics['min_entropy']:.4f}")
            st.write(f"Status: **{metrics['min_category']}**")

            st.metric("Collision Entropy", f"{metrics['collision']:.4f}")
            st.write(f"Status: **{metrics['collision_category']}**")
        with panel3:
            st.subheader("Panel 3: Statistical Tests")
            st.metric("Frequency Test", pass_fail_p_value(metrics["frequency_p"]))
            st.write(f"p-value: `{metrics['frequency_p']:.4f}`")
            st.write(f"Status: **{metrics['frequency_category']}**")

            st.metric("Runs Test", pass_fail_p_value(metrics["runs_p"]))
            st.write(f"p-value: `{metrics['runs_p']:.4f}`")
            st.write(f"Status: **{metrics['runs_category']}**")

            st.metric("Autocorrelation", pass_fail_autocorr(metrics["autocorrelation"]))
            st.write(f"Value: `{metrics['autocorrelation']:.4f}`")
            st.write(f"Status: **{metrics['autocorr_category']}**")
        with panel4:
            st.subheader("Panel 4: Degradation")
            st.metric("Bias", f"{metrics['bias'] * 100:.2f}%")
            st.write(f"Status: **{metrics['bias_category']}**")

            st.metric("Noise", f"{metrics['noise'] * 100:.2f}%")
            st.write(f"Status: **{metrics['noise_category']}**")

            st.metric("Drift", f"{metrics['drift'] * 100:.2f}%")
            st.write(f"Status: **{metrics['drift_category']}**")
        st.subheader("2-Qubit Output Counts")
        counts_df = pd.DataFrame(
            [{"Outcome": outcome, "Count": count} for outcome, count in sorted(counts.items())]
        )
        st.dataframe(counts_df, use_container_width=True, hide_index=True)
        st.subheader("Separate Metric Status Summary")

        entropy_status_df = pd.DataFrame(
            [
                ["Shannon Entropy", f"{metrics['shannon']:.4f}", metrics["shannon_category"]],
                ["Min Entropy", f"{metrics['min_entropy']:.4f}", metrics["min_category"]],
                ["Collision Entropy", f"{metrics['collision']:.4f}", metrics["collision_category"]],
            ],
            columns=["Entropy Metric", "Value", "Status"],
        )

        test_status_df = pd.DataFrame(
            [
                [
                    "Frequency Test",
                    f"p = {metrics['frequency_p']:.4f}",
                    pass_fail_p_value(metrics["frequency_p"]),
                    metrics["frequency_category"],
                ],
                [
                    "Runs Test",
                    f"p = {metrics['runs_p']:.4f}",
                    pass_fail_p_value(metrics["runs_p"]),
                    metrics["runs_category"],
                ],
                [
                    "Autocorrelation",
                    f"{metrics['autocorrelation']:.4f}",
                    pass_fail_autocorr(metrics["autocorrelation"]),
                    metrics["autocorr_category"],
                ],
            ],
            columns=["Statistical Test", "Value", "Result", "Status"],
        )

        degradation_status_df = pd.DataFrame(
            [
                ["Bias", f"{metrics['bias'] * 100:.2f}%", metrics["bias_category"]],
                ["Noise", f"{metrics['noise'] * 100:.2f}%", metrics["noise_category"]],
                ["Drift", f"{metrics['drift'] * 100:.2f}%", metrics["drift_category"]],
            ],
            columns=["Degradation Metric", "Value", "Status"],
        )

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.write("Entropy Status")
            st.dataframe(entropy_status_df, use_container_width=True, hide_index=True)

        with summary_col2:
            st.write("Statistical Test Status")
            st.dataframe(test_status_df, use_container_width=True, hide_index=True)

        with summary_col3:
            st.write("Degradation Status")
            st.dataframe(degradation_status_df, use_container_width=True, hide_index=True)
            st.subheader("Performance Improvement and Comparison")
        st.caption("Comparison of raw QRNG, extractor-improved QRNG, and classical PRNG.")

        comparison_df = build_comparison_table(bits)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        entropy_chart_df = comparison_df[
            ["Source", "Shannon", "Min Entropy", "Collision"]
        ].set_index("Source")

        ehi_chart_df = comparison_df[
            ["Source", "EHI (%)"]
        ].set_index("Source")

        st.write("Entropy Comparison Graph")
        st.bar_chart(entropy_chart_df, height=320)

        st.write("EHI Comparison Graph")
        st.bar_chart(ehi_chart_df, height=320)

        
 
    with graph_panel.container():
        st.subheader("Panel 5: Live Graph - Time vs EHI")
        graph_df = pd.DataFrame(
            {
                "Time (seconds)": st.session_state.time_history,
                "EHI": st.session_state.ehi_history,
            }
        )

        if len(graph_df) > 0:
            graph_df["Status"] = graph_df["EHI"].apply(final_status)

            threshold_bands = pd.DataFrame(
                [
                    {"y1": 0, "y2": 40, "Status": "Worst", "Color": "#f8d7da"},
                    {"y1": 40, "y2": 60, "Status": "Warning", "Color": "#ffe3e3"},
                    {"y1": 60, "y2": 75, "Status": "Average", "Color": "#fff3bf"},
                    {"y1": 75, "y2": 90, "Status": "Good", "Color": "#d3f9d8"},
                    {"y1": 90, "y2": 100, "Status": "Excellent", "Color": "#b2f2bb"},
                ]
            )

            chart_spec = {
                "height": 360,
                "layer": [
                    {
                        "data": {"values": threshold_bands.to_dict("records")},
                        "mark": {"type": "rect", "opacity": 0.55},
                        "encoding": {
                            "y": {"field": "y1", "type": "quantitative", "scale": {"domain": [0, 100]}},
                            "y2": {"field": "y2"},
                            "color": {
                                "field": "Status",
                                "type": "nominal",
                                "scale": {
                                    "domain": ["Worst", "Warning", "Average", "Good", "Excellent"],
                                    "range": ["#f8d7da", "#ffe3e3", "#fff3bf", "#d3f9d8", "#b2f2bb"],
                                },
                            },
                        },
                    },
                    {
                        "data": {"values": graph_df.to_dict("records")},
                        "mark": {"type": "line", "point": True, "strokeWidth": 3, "color": "#1c7ed6"},
                        "encoding": {
                            "x": {"field": "Time (seconds)", "type": "quantitative", "title": "Time (seconds)"},
                            "y": {"field": "EHI", "type": "quantitative", "title": "EHI (%)", "scale": {"domain": [0, 100]}},
                            "tooltip": [
                                {"field": "Time (seconds)", "type": "quantitative"},
                                {"field": "EHI", "type": "quantitative", "format": ".2f"},
                                {"field": "Status", "type": "nominal"},
                            ],
                        },
                    },
                    {
                        "data": {
                            "values": [
                                {"Threshold": "Warning", "EHI": 40},
                                {"Threshold": "Average", "EHI": 60},
                                {"Threshold": "Good", "EHI": 75},
                                {"Threshold": "Excellent", "EHI": 90},
                            ]
                        },
                        "mark": {"type": "rule", "strokeDash": [6, 4], "color": "#495057"},
                        "encoding": {
                            "y": {"field": "EHI", "type": "quantitative"},
                            "tooltip": [{"field": "Threshold", "type": "nominal"}, {"field": "EHI", "type": "quantitative"}],
                        },
                    },
                ],
                "resolve": {"scale": {"color": "independent"}},
            }

            st.vega_lite_chart(graph_df, chart_spec, use_container_width=True)

            st.subheader("Separate EHI Status Panel")
            ehi_status_df = pd.DataFrame(
                [
                    {
                        "Cycle": index + 1,
                        "Time (seconds)": st.session_state.time_history[index],
                        "EHI (%)": f"{ehi:.2f}",
                        "Status": final_status(ehi),
                    }
                    for index, ehi in enumerate(st.session_state.ehi_history)
                ]
            )
            st.dataframe(ehi_status_df, use_container_width=True, hide_index=True)
        else:
            st.info("Live graph will appear after generating readings.")


def generate_and_update():
    bits, counts = generate_qrng_bits(shots)
    if enable_degradation:
        bits = apply_bias(bits, bias_probability)
        bits = apply_noise(bits, noise_level)
        bits = apply_drift(bits, drift_strength)
    metrics, st.session_state.previous_bias = calculate_all_metrics(
        bits,
        st.session_state.previous_bias,
    )

    current_time = st.session_state.cycle_count * delay
    st.session_state.time_history.append(current_time)
    st.session_state.ehi_history.append(metrics["ehi"])
    st.session_state.cycle_count += 1

    render_dashboard(metrics, counts, bits)



if one_reading:
    generate_and_update()

if start_live:
    for _ in range(updates):
        generate_and_update()
        time.sleep(delay)

if not one_reading and not start_live:
    st.info("Use the sidebar to generate one reading or start real-time monitoring.")


# ============================================================
# 7. THRESHOLD TABLE + WEIGHTAGE TABLE
# ============================================================


# with threshold_panel.container():
#     st.divider()
#     st.subheader("Threshold Design")

#     threshold_data = [
#         ["Shannon Entropy", "< 0.900", "0.900-0.939", "0.940-0.969", "0.970-0.989", ">= 0.990"],
#         ["Min Entropy", "< 0.850", "0.850-0.899", "0.900-0.939", "0.940-0.969", ">= 0.970"],
#         ["Collision Entropy", "< 0.880", "0.880-0.929", "0.930-0.959", "0.960-0.979", ">= 0.980"],
#         ["Frequency Test p-value", "< 0.010", "0.010-0.029", "0.030-0.049", "0.050-0.099", ">= 0.100"],
#         ["Runs Test p-value", "< 0.010", "0.010-0.029", "0.030-0.049", "0.050-0.099", ">= 0.100"],
#         ["Autocorrelation", "> 0.100", "0.051-0.100", "0.021-0.050", "0.011-0.020", "<= 0.010"],
#         ["Bias", "> 5.0%", "2.1%-5.0%", "1.1%-2.0%", "0.6%-1.0%", "<= 0.5%"],
#         ["Noise", "> 5.0%", "2.1%-5.0%", "1.1%-2.0%", "0.6%-1.0%", "<= 0.5%"],
#         ["Drift", "> 5.0%", "2.1%-5.0%", "1.1%-2.0%", "0.6%-1.0%", "<= 0.5%"],
#     ]

#     threshold_df = pd.DataFrame(
#         threshold_data,
#         columns=["Component", "Worst", "Warning", "Average", "Good", "Excellent"],
#     )
#     st.dataframe(threshold_df, use_container_width=True, hide_index=True)

#     st.subheader("EHI Weightage Formula")

#     weight_data = [
#         ["Shannon Entropy", "Entropy", "12%"],
#         ["Min Entropy", "Entropy", "20%"],
#         ["Collision Entropy", "Entropy", "13%"],
#         ["Frequency Test", "Statistical Test", "10%"],
#         ["Runs Test", "Statistical Test", "10%"],
#         ["Autocorrelation Test", "Statistical Test", "10%"],
#         ["Bias", "Degradation", "10%"],
#         ["Noise", "Degradation", "7%"],
#         ["Drift", "Degradation", "8%"],
#     ]

#     weight_df = pd.DataFrame(weight_data, columns=["Metric", "Section", "Weight"])
#     st.dataframe(weight_df, use_container_width=True, hide_index=True)

#     st.code(
#         """
# EHI =
# 0.12 * Shannon_Score +
# 0.20 * MinEntropy_Score +
# 0.13 * CollisionEntropy_Score +
# 0.10 * FrequencyTest_Score +
# 0.10 * RunsTest_Score +
# 0.10 * Autocorrelation_Score +
# 0.10 * Bias_Score +
# 0.07 * Noise_Score +
# 0.08 * Drift_Score
#         """,
#         language="text",
#     )
