from base64 import b64encode
from datetime import datetime
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter


st.set_page_config(
    page_title="GreenVest | Sustainable Investing Workspace",
    layout="wide",
    initial_sidebar_state="collapsed",
)


ROOT = Path(__file__).parent
STATIC_DIR = ROOT / "static"
LOGO_PATH = STATIC_DIR / "Sustainable growth and innovation logo.png"
GLASSES_PATH = STATIC_DIR / "glasses.png"
QUESTION_PATH = STATIC_DIR / "question.png"

SECTORS = [
    "Clean Energy",
    "Technology",
    "Healthcare",
    "Financial Services",
    "Consumer Goods",
    "Industrials",
    "Real Estate",
    "Tobacco",
    "Weapons / Defence",
    "Gambling",
    "Fossil Fuels / Energy",
    "Alcohol",
]

REVIEWS = [
    {
        "author": "Blue Hoirzon",
        "quote": (
            "We absolutely love GreenVest. Our own app is striving to become like "
            "this one because the experience feels polished, trustworthy, and easy to follow."
        ),
    },
    {
        "author": "Octavian",
        "quote": (
            "The balance between sustainability and returns is explained so clearly "
            "that I can make decisions with confidence."
        ),
    },
    {
        "author": "Marget",
        "quote": (
            "Beautifully presented and surprisingly simple to use. The charts finally "
            "feel investor-friendly instead of academic."
        ),
    },
    {
        "author": "Anonymous Investor",
        "quote": (
            "I like how GreenVest keeps the important information front and center. "
            "It feels calm, modern, and professional."
        ),
    },
    {
        "author": "Northlight Capital",
        "quote": (
            "The onboarding flow makes sustainable investing feel approachable without "
            "oversimplifying the strategy."
        ),
    },
    {
        "author": "Willow Ridge",
        "quote": (
            "GreenVest turns a complicated decision into something clear and motivating. "
            "That is rare in finance tools."
        ),
    },
]


def image_to_data_uri(path: Path, make_transparent=False) -> str:
    if make_transparent:
        image = Image.open(path).convert("RGBA")
        pixels = np.array(image)
        whiteish = (
            (pixels[:, :, 0] > 240)
            & (pixels[:, :, 1] > 240)
            & (pixels[:, :, 2] > 240)
        )
        pixels[:, :, 3] = np.where(whiteish, 0, pixels[:, :, 3])
        cleaned = Image.fromarray(pixels)
        buffer = BytesIO()
        cleaned.save(buffer, format="PNG")
        encoded = b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


LOGO_DATA_URI = image_to_data_uri(LOGO_PATH, make_transparent=True)
GLASSES_DATA_URI = image_to_data_uri(GLASSES_PATH, make_transparent=True)
QUESTION_DATA_URI = image_to_data_uri(QUESTION_PATH, make_transparent=True)


def get_query_param(key: str):
    try:
        value = st.query_params.get(key)
        if isinstance(value, list):
            return value[0] if value else None
        return value
    except Exception:
        return st.experimental_get_query_params().get(key, [None])[0]


def clear_query_params():
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()


def inject_styles():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');

            :root {
                --green-900: #163a2a;
                --green-700: #246847;
                --green-600: #2e7b53;
                --green-500: #46a469;
                --green-200: #d8ecde;
                --green-100: #edf7f0;
                --blue-600: #2f7dca;
                --blue-500: #54a8e8;
                --blue-100: #edf5fc;
                --ink-900: #183427;
                --ink-700: #406854;
                --ink-500: #6a8a77;
                --card: rgba(255, 255, 255, 0.88);
                --border: rgba(46, 123, 83, 0.16);
                --shadow: 0 24px 60px rgba(26, 68, 47, 0.10);
            }

            .stApp,
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(94, 174, 119, 0.16), transparent 28%),
                    radial-gradient(circle at top right, rgba(76, 164, 224, 0.14), transparent 24%),
                    linear-gradient(180deg, #f4fbf6 0%, #edf7f1 34%, #f7fbf8 100%);
                color: var(--ink-900);
                font-family: "Manrope", "Trebuchet MS", sans-serif;
            }

            .block-container {
                padding-top: 1.1rem;
                padding-bottom: 4.5rem;
                max-width: 1220px;
            }

            h1, h2, h3, h4, h5 {
                font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
                color: var(--green-900);
                letter-spacing: -0.02em;
            }

            p, li, label, [data-testid="stMarkdownContainer"] {
                color: var(--ink-700);
            }

            .loader-overlay {
                position: fixed;
                inset: 0;
                z-index: 9999;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 0.95rem;
                background:
                    radial-gradient(circle at center, rgba(132, 213, 154, 0.42), transparent 35%),
                    linear-gradient(180deg, #f7fcf8 0%, #eff7f1 100%);
                animation: loader-fade 2.3s ease forwards;
            }

            .loader-logo {
                width: 118px;
                height: 118px;
                object-fit: contain;
                filter: drop-shadow(0 16px 26px rgba(27, 92, 57, 0.16));
                animation: loader-float 1.2s ease-in-out infinite alternate;
            }

            .loader-title {
                font-family: "Space Grotesk", "Trebuchet MS", sans-serif;
                font-size: 2.3rem;
                font-weight: 700;
                color: var(--green-900);
            }

            .loader-copy {
                font-size: 0.98rem;
                font-weight: 600;
                color: var(--ink-700);
                letter-spacing: 0.02em;
            }

            @keyframes loader-fade {
                0%, 62% { opacity: 1; visibility: visible; }
                100% { opacity: 0; visibility: hidden; pointer-events: none; }
            }

            @keyframes loader-float {
                from { transform: translateY(-4px) scale(1); }
                to { transform: translateY(4px) scale(1.04); }
            }

            .hero-shell,
            .glass-card,
            .review-shell,
            .dashboard-hero,
            .spotlight-card,
            .profile-banner {
                background: var(--card);
                backdrop-filter: blur(10px);
                border: 1px solid var(--border);
                box-shadow: var(--shadow);
                border-radius: 28px;
            }

            .hero-shell {
                padding: 2.4rem 2.5rem;
                overflow: hidden;
                position: relative;
                margin-bottom: 1.4rem;
            }

            .hero-shell::before {
                content: "";
                position: absolute;
                inset: 0 auto auto 0;
                width: 100%;
                height: 10px;
                background: linear-gradient(90deg, var(--green-700), var(--blue-500));
            }

            .hero-grid {
                display: grid;
                grid-template-columns: 1.3fr 0.9fr;
                gap: 1.2rem;
                align-items: stretch;
            }

            .hero-brand {
                display: flex;
                gap: 1rem;
                align-items: flex-start;
                margin-top: 0.6rem;
                margin-bottom: 0.75rem;
            }

            .hero-logo {
                width: 86px;
                height: 86px;
                object-fit: contain;
                filter: drop-shadow(0 10px 20px rgba(31, 88, 58, 0.16));
                margin-top: -0.45rem;
                flex-shrink: 0;
            }

            .hero-kicker,
            .section-kicker {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 0.74rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: var(--green-700);
            }

            .hero-kicker::before,
            .section-kicker::before {
                content: "";
                width: 12px;
                height: 12px;
                border-radius: 999px;
                background: linear-gradient(135deg, var(--green-500), var(--blue-500));
            }

            .hero-title {
                font-size: clamp(2.9rem, 5vw, 4.2rem);
                margin: 0.2rem 0 0.5rem;
                line-height: 0.95;
            }

            .hero-copy {
                font-size: 1.05rem;
                line-height: 1.8;
                max-width: 56ch;
                margin-bottom: 1.35rem;
            }

            .hero-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 0.95rem;
                margin-bottom: 0.75rem;
            }

            .hero-button {
                text-decoration: none !important;
                color: white !important;
                font-weight: 800;
                padding: 0.95rem 1.45rem;
                border-radius: 18px;
                min-width: 230px;
                text-align: center;
                box-shadow: 0 20px 30px rgba(37, 100, 68, 0.16);
                border: 1px solid rgba(255, 255, 255, 0.46);
            }

            .hero-button.green {
                background: linear-gradient(135deg, #257449 0%, #4eb772 100%);
            }

            .hero-button.blue {
                background: linear-gradient(135deg, #2d76cf 0%, #5cb8ef 100%);
            }

            .hero-button:hover {
                transform: translateY(-1px);
            }

            .hero-note {
                font-size: 0.92rem;
                color: var(--ink-500);
                font-weight: 600;
            }

            .hero-panel {
                background: linear-gradient(180deg, rgba(235, 247, 238, 0.9), rgba(238, 246, 252, 0.92));
                border-radius: 24px;
                border: 1px solid rgba(46, 123, 83, 0.12);
                padding: 1.4rem;
                position: relative;
                overflow: hidden;
                min-height: 100%;
            }

            .hero-panel h4 {
                margin: 0 0 0.55rem;
                font-size: 1.12rem;
            }

            .hero-panel-copy {
                max-width: calc(100% - 140px);
            }

            .hero-points {
                display: grid;
                gap: 0.75rem;
                margin-top: 1rem;
            }

            .hero-point {
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid rgba(46, 123, 83, 0.11);
                border-radius: 18px;
                padding: 0.9rem 1rem;
            }

            .hero-point strong {
                display: block;
                color: var(--green-900);
                margin-bottom: 0.2rem;
            }

            .hero-mascot {
                position: absolute;
                right: -0.4rem;
                bottom: -0.85rem;
                width: 155px;
                filter: drop-shadow(0 14px 24px rgba(24, 60, 43, 0.12));
            }

            .fact-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 1rem;
                margin: 1.35rem 0 1.1rem;
            }

            .fact-card {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(46, 123, 83, 0.12);
                border-radius: 22px;
                padding: 1.15rem 1.2rem;
                min-height: 170px;
            }

            .fact-card h4 {
                margin: 0 0 0.6rem;
                font-size: 1.08rem;
            }

            .fact-card p {
                margin: 0;
                line-height: 1.75;
                font-size: 0.98rem;
            }

            .review-shell {
                padding: 1.4rem 1.55rem 1.6rem;
                overflow: hidden;
                position: relative;
                margin-top: 1.1rem;
            }

            .review-intro {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: end;
                margin-bottom: 1rem;
            }

            .review-intro h3 {
                margin: 0.2rem 0 0;
                font-size: 1.7rem;
            }

            .review-rail {
                overflow: hidden;
                mask-image: linear-gradient(to right, transparent, black 12%, black 88%, transparent);
                -webkit-mask-image: linear-gradient(to right, transparent, black 12%, black 88%, transparent);
            }

            .review-track {
                display: flex;
                gap: 1rem;
                width: max-content;
                animation: reviews-left-to-right 30s linear infinite;
            }

            .review-card {
                width: 320px;
                min-height: 190px;
                padding: 1.15rem 1.2rem;
                border-radius: 22px;
                border: 1px solid rgba(46, 123, 83, 0.12);
                background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(240,248,243,0.94));
                box-shadow: 0 18px 28px rgba(24, 60, 43, 0.08);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .review-stars {
                color: #f3b31d;
                letter-spacing: 0.08em;
                font-size: 0.9rem;
                margin-bottom: 0.65rem;
            }

            .review-card p {
                color: var(--ink-700);
                line-height: 1.68;
                font-size: 0.95rem;
                margin: 0 0 0.9rem;
            }

            .reviewer {
                font-weight: 800;
                color: var(--green-900);
                font-size: 0.95rem;
            }

            @keyframes reviews-left-to-right {
                0% { transform: translateX(-52%); }
                100% { transform: translateX(0); }
            }

            .dashboard-hero {
                padding: 1.75rem 1.85rem;
                margin-bottom: 1.35rem;
            }

            .dashboard-head {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: start;
                flex-wrap: wrap;
            }

            .dashboard-title {
                font-size: 2.2rem;
                margin: 0.25rem 0 0.45rem;
            }

            .dashboard-copy {
                max-width: 58ch;
                line-height: 1.75;
                margin: 0;
                color: var(--ink-700);
            }

            .chip-row {
                display: flex;
                gap: 0.6rem;
                flex-wrap: wrap;
                margin-top: 1rem;
            }

            .chip {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.52rem 0.82rem;
                border-radius: 999px;
                font-size: 0.84rem;
                font-weight: 700;
                border: 1px solid rgba(46, 123, 83, 0.14);
                background: rgba(255, 255, 255, 0.85);
                color: var(--green-900);
            }

            .chip.blue {
                border-color: rgba(47, 125, 202, 0.18);
                color: #235f9b;
                background: rgba(240, 247, 255, 0.95);
            }

            .profile-banner {
                padding: 1.35rem 1.45rem;
                margin: 0.7rem 0 1.4rem;
            }

            .guide-mascot-card {
                background: linear-gradient(180deg, rgba(241, 248, 244, 0.98), rgba(234, 244, 252, 0.96));
                border: 1px solid rgba(46, 123, 83, 0.12);
                border-radius: 24px;
                padding: 1.15rem 1.1rem 1rem;
                box-shadow: 0 16px 28px rgba(24, 60, 43, 0.06);
            }

            .guide-mascot-card h4 {
                margin: 0.25rem 0 0.45rem;
                font-size: 1.2rem;
            }

            .guide-mascot-card p {
                margin: 0;
                line-height: 1.72;
                font-size: 0.95rem;
            }

            .guide-mascot-image {
                display: block;
                width: min(220px, 100%);
                margin: 0.9rem auto 0;
                filter: drop-shadow(0 12px 22px rgba(24, 60, 43, 0.12));
            }

            .spotlight-card {
                padding: 1.45rem 1.4rem;
                margin-bottom: 1rem;
                background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(237,247,241,0.92));
            }

            .spotlight-card h3 {
                margin: 0.3rem 0 0.55rem;
                font-size: 1.45rem;
            }

            .spotlight-card p {
                margin: 0;
                line-height: 1.74;
            }

            div[data-testid="metric-container"] {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(46, 123, 83, 0.12);
                border-radius: 18px;
                padding: 1rem 1.05rem;
                box-shadow: 0 12px 24px rgba(24, 60, 43, 0.06);
            }

            [data-testid="metric-container"] label {
                color: var(--ink-500);
            }

            .section-title {
                margin: 0.35rem 0 0.7rem;
                font-size: 1.75rem;
            }

            .section-copy {
                margin: 0;
                line-height: 1.72;
                color: var(--ink-700);
            }

            .stButton > button {
                width: 100%;
                border-radius: 16px;
                border: 1px solid rgba(46, 123, 83, 0.12);
                background: linear-gradient(135deg, #256e45, #49a96b);
                color: white;
                font-weight: 700;
                padding: 0.7rem 1rem;
                box-shadow: 0 14px 22px rgba(37, 100, 68, 0.12);
            }

            .stButton > button:hover {
                color: white !important;
                border-color: rgba(46, 123, 83, 0.16);
            }

            .info-note {
                padding: 1rem 1.05rem;
                border-radius: 18px;
                background: rgba(236, 247, 240, 0.92);
                border: 1px solid rgba(46, 123, 83, 0.12);
                line-height: 1.7;
                margin: 0.8rem 0 0.3rem;
            }

            .stExpander {
                border: 1px solid rgba(46, 123, 83, 0.12) !important;
                border-radius: 18px !important;
                background: rgba(255, 255, 255, 0.88) !important;
            }

            .stExpander details summary p {
                font-weight: 700;
                color: var(--green-900);
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="input"] > div,
            textarea,
            input {
                border-radius: 14px !important;
            }

            .stSlider [data-baseweb="slider"] {
                padding-top: 0.4rem;
                padding-bottom: 0.1rem;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid rgba(46, 123, 83, 0.12);
                border-radius: 18px;
                overflow: hidden;
                box-shadow: 0 12px 24px rgba(24, 60, 43, 0.06);
            }

            .chart-card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid rgba(46, 123, 83, 0.12);
                border-radius: 24px;
                padding: 1rem 1rem 0.2rem;
                box-shadow: 0 20px 35px rgba(24, 60, 43, 0.07);
            }

            .chart-caption {
                font-size: 0.92rem;
                color: var(--ink-500);
                margin-top: -0.4rem;
                margin-bottom: 0.8rem;
            }

            @media (max-width: 980px) {
                .hero-grid {
                    grid-template-columns: 1fr;
                }

                .fact-grid {
                    grid-template-columns: 1fr;
                }

                .hero-button {
                    width: 100%;
                    min-width: 0;
                }

                .hero-panel-copy {
                    max-width: 100%;
                }

                .hero-mascot {
                    position: static;
                    width: 120px;
                    margin: 1rem 0 0 auto;
                    display: block;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_loader():
    st.markdown(
        f"""
        <div class="loader-overlay">
            <img class="loader-logo" src="{LOGO_DATA_URI}" alt="GreenVest logo">
            <div class="loader-title">GreenVest</div>
            <div class="loader-copy">Loading your sustainable investing workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def p_return(w1, mu1, mu2):
    return w1 * mu1 + (1 - w1) * mu2


def p_std(w1, s1, s2, rho):
    return np.sqrt(w1**2 * s1**2 + (1 - w1) ** 2 * s2**2 + 2 * rho * w1 * (1 - w1) * s1 * s2)


def p_esg(w1, e1, e2):
    return w1 * e1 + (1 - w1) * e2


def p_util(mu, sig, esg, gamma, lam):
    return mu - (gamma / 2) * sig**2 + lam * esg


def composite_esg(e, s, g, we, ws, wg):
    total_weight = we + ws + wg
    return (we * e + ws * s + wg * g) / total_weight if total_weight > 0 else 0


def esg_momentum(now, last):
    return (now - last) / last if last > 0 else 0


def esg_rating(score):
    for thresh, rating in [
        (0.85, "AAA"),
        (0.70, "AA"),
        (0.55, "A"),
        (0.40, "BBB"),
        (0.25, "BB"),
        (0.10, "B"),
    ]:
        if score >= thresh:
            return rating
    return "CCC"


def esg_sharpe(mu, rf, sig, lam, esg):
    return (mu - rf + lam * esg) / sig if sig > 0 else float("nan")


def future_value(pv, r, years):
    return pv * (1 + r) ** years


def optimise(mu1, mu2, s1, s2, e1, e2, rho, gamma, lam, n=2000, force_w1=None):
    if force_w1 is not None:
        weights = np.array([force_w1])
        mu_grid = p_return(weights, mu1, mu2)
        sigma_grid = p_std(weights, s1, s2, rho)
        esg_grid = p_esg(weights, e1, e2)
        utility_grid = p_util(mu_grid, sigma_grid, esg_grid, gamma, lam)
        return weights, mu_grid, sigma_grid, esg_grid, utility_grid, mu_grid + lam * esg_grid, 0

    weights = np.linspace(0, 1, n)
    mu_grid = p_return(weights, mu1, mu2)
    sigma_grid = p_std(weights, s1, s2, rho)
    esg_grid = p_esg(weights, e1, e2)
    utility_grid = p_util(mu_grid, sigma_grid, esg_grid, gamma, lam)
    idx = np.argmax(utility_grid)
    return weights, mu_grid, sigma_grid, esg_grid, utility_grid, mu_grid + lam * esg_grid, idx


def derive_lambda(e_w, s_w, g_w, excl_count, goal_lam):
    base = ((e_w + s_w + g_w) / 3 / 5) * 0.11 + 0.01
    lam = min(base + 0.005 * excl_count + goal_lam, 0.20)
    return round(lam, 3)


def style_axis(ax):
    ax.set_facecolor("#ffffff")
    ax.grid(True, color="#d6e7d9", linewidth=0.85, alpha=0.75)
    for spine in ax.spines.values():
        spine.set_color("#cddfd2")
    ax.tick_params(colors="#406854", labelsize=10)
    ax.xaxis.label.set_color("#406854")
    ax.yaxis.label.set_color("#406854")


def build_frontier_chart(
    sigma_grid,
    esg_adjusted_grid,
    esg_grid,
    sigma_opt,
    esg_adjusted_opt,
    sigma_financial,
    esg_adjusted_financial,
    sigma_low_risk,
    esg_adjusted_low_risk,
):
    fig, ax = plt.subplots(figsize=(8.2, 5.7))
    fig.patch.set_facecolor("#ffffff")
    style_axis(ax)

    points = np.array([sigma_grid, esg_adjusted_grid]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    collection = LineCollection(segments, cmap="YlGn", linewidth=4)
    collection.set_array(esg_grid)
    ax.add_collection(collection)
    ax.autoscale()

    ax.plot(
        sigma_grid,
        esg_adjusted_grid,
        color="#afcdb7",
        linestyle="--",
        linewidth=1.3,
        alpha=0.9,
        label="Traditional frontier path",
    )
    ax.scatter(
        sigma_opt,
        esg_adjusted_opt,
        s=72,
        color="#1f6844",
        edgecolors="#d8f2e0",
        linewidths=1.4,
        label="GreenVest optimum",
        zorder=5,
    )
    ax.scatter(
        sigma_financial,
        esg_adjusted_financial,
        s=54,
        color="#2f7dca",
        edgecolors="#e8f3fd",
        linewidths=1.1,
        label="Financial-only benchmark",
        zorder=4,
    )
    ax.scatter(
        sigma_low_risk,
        esg_adjusted_low_risk,
        s=54,
        color="#5d9b79",
        edgecolors="#edf7f0",
        linewidths=1.1,
        label="Lowest-risk mix",
        zorder=4,
    )

    colorbar = fig.colorbar(collection, ax=ax, pad=0.02, shrink=0.88)
    colorbar.set_label("ESG score", color="#406854")
    colorbar.ax.tick_params(labelsize=9, colors="#406854")

    ax.set_title("ESG Efficient Frontier", fontsize=15, fontweight="bold", color="#163a2a", pad=14)
    ax.set_xlabel("Portfolio risk (std deviation)")
    ax.set_ylabel("ESG-adjusted return")
    ax.legend(
        loc="lower right",
        fontsize=8.6,
        frameon=True,
        facecolor="white",
        edgecolor="#d6e7d9",
        framealpha=0.95,
        markerscale=0.8,
        borderpad=0.55,
        labelspacing=0.55,
        handlelength=1.4,
    )
    fig.tight_layout(pad=1.4)
    return fig


def build_future_value_chart(invest, opt_return, benchmark_return=None, selected_label="GreenVest"):
    years = np.arange(0, 31)
    fig, ax = plt.subplots(figsize=(7.8, 5.7))
    fig.patch.set_facecolor("#ffffff")
    style_axis(ax)

    opt_values = [future_value(invest, opt_return, year) for year in years]
    ax.plot(
        years,
        opt_values,
        color="#1f6844",
        linewidth=3.2,
        label=f"{selected_label} strategy",
    )
    ax.fill_between(years, opt_values, color="#d9efdf", alpha=0.4)

    if benchmark_return is not None:
        benchmark_values = [future_value(invest, benchmark_return, year) for year in years]
        ax.plot(
            years,
            benchmark_values,
            color="#2f7dca",
            linewidth=2.4,
            linestyle="--",
            label="Balanced 50 / 50 reference",
        )

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"GBP {x:,.0f}"))
    ax.set_xlabel("Years invested")
    ax.set_ylabel("Projected portfolio value")
    ax.set_title("Future Value Projection", fontsize=15, fontweight="bold", color="#163a2a", pad=14)
    ax.legend(
        loc="upper left",
        fontsize=8.8,
        frameon=True,
        facecolor="white",
        edgecolor="#d6e7d9",
        framealpha=0.95,
        borderpad=0.55,
        labelspacing=0.55,
        handlelength=1.8,
    )
    fig.tight_layout(pad=1.4)
    return fig


def build_summary_pdf_bytes(
    asset_one,
    asset_two,
    results,
    invest,
    profile,
    goal_label,
    gamma_used,
    lambda_used,
    e_w,
    s_w,
    g_w,
    excluded_summary,
):
    buffer = BytesIO()
    years = np.arange(0, 31)
    future_values = [future_value(invest, results["mu_opt"], year) for year in years]
    benchmark_values = None
    if results["benchmark_return"] is not None:
        benchmark_values = [future_value(invest, results["benchmark_return"], year) for year in years]

    checkpoints = [5, 10, 20, 30]
    checkpoint_lines = [
        f"{year}y: GBP {future_value(invest, results['mu_opt'], year):,.0f}"
        for year in checkpoints
    ]

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27))
        fig.patch.set_facecolor("#f7fbf8")
        grid = fig.add_gridspec(
            2,
            3,
            height_ratios=[0.88, 1.12],
            width_ratios=[1.05, 1.1, 1.05],
            hspace=0.28,
            wspace=0.28,
        )

        ax_header = fig.add_subplot(grid[0, :])
        ax_alloc = fig.add_subplot(grid[1, 0])
        ax_growth = fig.add_subplot(grid[1, 1])
        ax_notes = fig.add_subplot(grid[1, 2])

        ax_header.axis("off")
        ax_header.text(
            0.0,
            0.96,
            "GreenVest - One-page Portfolio Summary",
            fontsize=23,
            fontweight="bold",
            color="#163a2a",
            transform=ax_header.transAxes,
        )
        ax_header.text(
            0.0,
            0.80,
            f"Prepared on {datetime.now().strftime('%B %d, %Y')} for investor meetings and submissions.",
            fontsize=11.2,
            color="#4f7461",
            transform=ax_header.transAxes,
        )
        ax_header.text(
            0.0,
            0.56,
            (
                f"Investor profile: {profile}   |   Goal: {goal_label}   |   "
                f"gamma: {gamma_used:.1f}   |   lambda: {lambda_used:.3f}"
            ),
            fontsize=11.4,
            color="#224d38",
            transform=ax_header.transAxes,
        )
        ax_header.text(
            0.0,
            0.41,
            (
                f"Recommended allocation: {asset_one['name']} {results['w1'] * 100:.1f}%   |   "
                f"{asset_two['name']} {results['w2'] * 100:.1f}%"
            ),
            fontsize=13.5,
            fontweight="bold",
            color="#173b2b",
            transform=ax_header.transAxes,
        )
        ax_header.text(
            0.0,
            0.22,
            (
                f"ESG priorities: E {e_w}/5, S {s_w}/5, G {g_w}/5   |   "
                f"Excluded sectors: {excluded_summary}"
            ),
            fontsize=10.8,
            color="#4f7461",
            transform=ax_header.transAxes,
        )
        ax_header.text(
            0.0,
            0.04,
            (
                "Method note: GreenVest combines weighted E, S, and G pillar scores, applies a small "
                "momentum adjustment for improving companies, and then blends ESG into the utility score."
            ),
            fontsize=10.4,
            color="#5b7d69",
            transform=ax_header.transAxes,
        )

        style_axis(ax_alloc)
        allocation_names = [asset_one["name"], asset_two["name"]]
        allocation_values = [results["w1"] * 100, results["w2"] * 100]
        allocation_colors = ["#2e7b53", "#2f7dca"]
        positions = np.arange(len(allocation_names))
        ax_alloc.barh(positions, allocation_values, color=allocation_colors, height=0.45)
        ax_alloc.set_yticks(positions, allocation_names)
        ax_alloc.set_xlim(0, 100)
        ax_alloc.set_xlabel("Portfolio weight (%)")
        ax_alloc.set_title("Recommended allocation", fontsize=13.5, fontweight="bold", color="#163a2a", pad=12)
        for idx, value in enumerate(allocation_values):
            ax_alloc.text(value + 2, idx, f"{value:.1f}%", va="center", fontsize=10.2, color="#224d38")

        style_axis(ax_growth)
        ax_growth.plot(years, future_values, color="#1f6844", linewidth=3.0, label="GreenVest strategy")
        ax_growth.fill_between(years, future_values, color="#d9efdf", alpha=0.34)
        if benchmark_values is not None:
            ax_growth.plot(
                years,
                benchmark_values,
                color="#2f7dca",
                linewidth=2.0,
                linestyle="--",
                label="Balanced 50 / 50 reference",
            )
        ax_growth.set_title("Future value outlook", fontsize=13.5, fontweight="bold", color="#163a2a", pad=12)
        ax_growth.set_xlabel("Years invested")
        ax_growth.set_ylabel("Projected value")
        ax_growth.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"GBP {x:,.0f}"))
        ax_growth.legend(
            loc="upper left",
            fontsize=7.8,
            frameon=True,
            facecolor="white",
            edgecolor="#d6e7d9",
            framealpha=0.96,
        )

        ax_notes.axis("off")
        ax_notes.text(0.0, 0.96, "Key portfolio facts", fontsize=14, fontweight="bold", color="#163a2a")
        note_lines = [
            f"Expected return: {results['mu_opt'] * 100:.2f}%",
            f"Portfolio risk: {results['sigma_opt'] * 100:.2f}%",
            f"Sharpe ratio: {results['sharpe_opt']:.3f}",
            f"ESG score: {results['esg_opt'] * 100:.1f} / 100 [{esg_rating(results['esg_opt'])}]",
            f"Impact score: {results['impact_score']:.1f}",
            f"Trade-off score: {results['tradeoff']:.4f}",
            "",
            "Projection checkpoints",
            *checkpoint_lines,
        ]
        y_pos = 0.86
        for line in note_lines:
            if line == "":
                y_pos -= 0.08
                continue
            style = dict(fontsize=10.8, color="#355843")
            if line == "Projection checkpoints":
                style = dict(fontsize=11.5, fontweight="bold", color="#163a2a")
            ax_notes.text(0.0, y_pos, line, transform=ax_notes.transAxes, **style)
            y_pos -= 0.085

        pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()


def get_excluded_sectors():
    mapping = [
        ("Tobacco", st.session_state.excl_tobacco),
        ("Weapons / Defence", st.session_state.excl_weapons),
        ("Gambling", st.session_state.excl_gambling),
        ("Fossil Fuels / Energy", st.session_state.excl_fossil),
        ("Alcohol", st.session_state.excl_alcohol),
    ]
    return {name for name, enabled in mapping if enabled}


def excluded_sector_labels():
    mapping = [
        ("Tobacco", st.session_state.excl_tobacco),
        ("Weapons", st.session_state.excl_weapons),
        ("Gambling", st.session_state.excl_gambling),
        ("Fossil Fuels", st.session_state.excl_fossil),
        ("Alcohol", st.session_state.excl_alcohol),
    ]
    labels = [name for name, enabled in mapping if enabled]
    return labels if labels else ["None"]


def sync_gamma_slider():
    st.session_state.gamma_val = st.session_state._g_sl


def sync_gamma_input():
    st.session_state.gamma_val = st.session_state._g_ni


def sync_lambda_slider():
    st.session_state.lambda_val = st.session_state._l_sl


def sync_lambda_input():
    st.session_state.lambda_val = st.session_state._l_ni


def apply_profile_results(e_w, s_w, g_w, excl_tobacco, excl_weapons, excl_gambling, excl_fossil, excl_alcohol):
    st.session_state.e_w = e_w
    st.session_state.s_w = s_w
    st.session_state.g_w = g_w
    st.session_state.excl_tobacco = excl_tobacco
    st.session_state.excl_weapons = excl_weapons
    st.session_state.excl_gambling = excl_gambling
    st.session_state.excl_fossil = excl_fossil
    st.session_state.excl_alcohol = excl_alcohol

    q1 = st.session_state.q1
    q2 = st.session_state.q2
    q3 = st.session_state.q3
    score = ((5 - q1) + (5 - q2) + q3) / 3
    gamma_base = round(2 + (score - 1) * (8 / 3), 1)
    profile = "Conservative" if gamma_base >= 7 else "Balanced" if gamma_base >= 4 else "Aggressive"
    goal = st.session_state.goal
    gamma = round(max(1.0, gamma_base + {1: 2, 2: 0, 3: 0, 4: -1}[goal]), 1)
    goal_lam = {1: 0, 2: 0, 3: 0.02, 4: 0}[goal]
    excl_count = sum([excl_tobacco, excl_weapons, excl_gambling, excl_fossil, excl_alcohol])
    lambda_esg = derive_lambda(e_w, s_w, g_w, excl_count, goal_lam)

    goal_labels = {
        1: "Retirement",
        2: "Long-term growth",
        3: "Ethical investing",
        4: "Short-term profit",
    }

    st.session_state.gamma = gamma
    st.session_state.gamma_val = gamma
    st.session_state.lambda_esg = lambda_esg
    st.session_state.lambda_val = lambda_esg
    st.session_state.profile = profile
    st.session_state.goal_label = goal_labels[goal]
    st.session_state.onboarding_done = True
    st.session_state.entered_app = True
    st.session_state.setup_mode = "guided"
    st.session_state.show_profile_builder = False
    st.session_state.onboarding_step = 1
    st.rerun()


def render_profile_builder(editing=False):
    total_steps = 5
    step = st.session_state.onboarding_step
    heading = "Investor Preference Builder" if not editing else "Update Investor Preferences"
    body = (
        "Answer five short questions and GreenVest will shape the portfolio around the investor's goals."
        if not editing
        else "Adjust the profile inline and the recommendation will refresh with the new sustainability preferences."
    )

    st.markdown(
        f"""
        <div class="profile-banner">
            <div class="section-kicker">Guided setup</div>
            <h3 class="section-title" style="margin-bottom:0.35rem;">{heading}</h3>
            <p class="section-copy">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    content_col, mascot_col = st.columns([1.18, 0.42], gap="large")

    with mascot_col:
        st.markdown(
            f"""
            <div class="guide-mascot-card">
                <div class="section-kicker">Investor guide</div>
                <h4>Preference helper</h4>
                <p>
                    This question mascot sits beside the profiling flow and keeps the focus on the
                    answers that shape risk, sustainability, and sector exclusions.
                </p>
                <img class="guide-mascot-image" src="{QUESTION_DATA_URI}" alt="Question mascot">
            </div>
            """,
            unsafe_allow_html=True,
        )

    with content_col:
        st.progress(step / total_steps)
        st.caption(f"Step {step} of {total_steps}")
        st.write("")

        if step == 1:
            choice = st.radio(
                "If your portfolio dropped 25%, what would you do?",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Sell everything - I cannot afford to lose more",
                    2: "Reduce exposure - I want to limit further losses",
                    3: "Hold steady - I trust the market will recover",
                    4: "Buy more - downturns are buying opportunities",
                }[x],
                index=st.session_state.q1 - 1,
            )
            st.caption("This tells GreenVest how much short-term volatility feels comfortable for the investor.")
            left, _, right = st.columns([1, 1.4, 1])
            if editing:
                with left:
                    if st.button("Cancel", key="profile_cancel_1"):
                        st.session_state.show_profile_builder = False
                        st.session_state.onboarding_step = 1
                        st.rerun()
            with right:
                if st.button("Next", key="profile_next_1"):
                    st.session_state.q1 = choice
                    st.session_state.onboarding_step = 2
                    st.rerun()

        elif step == 2:
            choice = st.radio(
                "What is the primary investment objective?",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Preserve capital - protect what is already there",
                    2: "Generate income - steady, reliable returns",
                    3: "Long-term growth - grow wealth steadily",
                    4: "Maximise growth - seek the highest return potential",
                }[x],
                index=st.session_state.q2 - 1,
            )
            st.caption("This shifts the balance between stability, growth, and sustainability tilt.")
            left, _, right = st.columns([1, 1.4, 1])
            with left:
                if st.button("Back", key="profile_back_2"):
                    st.session_state.q2 = choice
                    st.session_state.onboarding_step = 1
                    st.rerun()
            with right:
                if st.button("Next", key="profile_next_2"):
                    st.session_state.q2 = choice
                    st.session_state.onboarding_step = 3
                    st.rerun()

        elif step == 3:
            choice = st.radio(
                "How long does the investor plan to stay invested?",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Less than 2 years",
                    2: "2 to 5 years",
                    3: "5 to 10 years",
                    4: "10 years or more",
                }[x],
                index=st.session_state.q3 - 1,
            )
            st.caption(
                "A longer horizon allows GreenVest to tolerate more short-term movement for better long-term upside."
            )
            left, _, right = st.columns([1, 1.4, 1])
            with left:
                if st.button("Back", key="profile_back_3"):
                    st.session_state.q3 = choice
                    st.session_state.onboarding_step = 2
                    st.rerun()
            with right:
                if st.button("Next", key="profile_next_3"):
                    st.session_state.q3 = choice
                    st.session_state.onboarding_step = 4
                    st.rerun()

        elif step == 4:
            choice = st.radio(
                "What is the investor working toward?",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Retirement - long-term financial security",
                    2: "Long-term growth - compounding wealth over time",
                    3: "Ethical investing - sustainability alongside returns",
                    4: "Short-term profit - stronger near-term gains",
                }[x],
                index=st.session_state.goal - 1,
            )
            st.caption("This final objective helps decide how much explicit weight ESG receives in the utility score.")
            left, _, right = st.columns([1, 1.4, 1])
            with left:
                if st.button("Back", key="profile_back_4"):
                    st.session_state.goal = choice
                    st.session_state.onboarding_step = 3
                    st.rerun()
            with right:
                if st.button("Next", key="profile_next_4"):
                    st.session_state.goal = choice
                    st.session_state.onboarding_step = 5
                    st.rerun()

        else:
            st.markdown("#### How much should GreenVest care about each ESG pillar?")
            st.caption("Set each pillar from 0 to 5. These weights shape the composite ESG score.")
            e_w = st.slider("Environmental priority", 0, 5, st.session_state.e_w)
            s_w = st.slider("Social priority", 0, 5, st.session_state.s_w)
            g_w = st.slider("Governance priority", 0, 5, st.session_state.g_w)
            st.write("")
            st.caption("Exclude any sectors from the portfolio universe?")
            c1, c2, c3, c4, c5 = st.columns(5)
            excl_tobacco = c1.checkbox("Tobacco", value=st.session_state.excl_tobacco)
            excl_weapons = c2.checkbox("Weapons", value=st.session_state.excl_weapons)
            excl_gambling = c3.checkbox("Gambling", value=st.session_state.excl_gambling)
            excl_fossil = c4.checkbox("Fossil Fuels", value=st.session_state.excl_fossil)
            excl_alcohol = c5.checkbox("Alcohol", value=st.session_state.excl_alcohol)

            left, _, right = st.columns([1, 1.4, 1])
            with left:
                if st.button("Back", key="profile_back_5"):
                    st.session_state.e_w = e_w
                    st.session_state.s_w = s_w
                    st.session_state.g_w = g_w
                    st.session_state.onboarding_step = 4
                    st.rerun()
            with right:
                if st.button("Build My GreenVest Profile", key="profile_done_5"):
                    apply_profile_results(
                        e_w,
                        s_w,
                        g_w,
                        excl_tobacco,
                        excl_weapons,
                        excl_gambling,
                        excl_fossil,
                        excl_alcohol,
                    )


def render_landing_page():
    review_cards = REVIEWS + REVIEWS
    review_html = "".join(
        f"""
        <div class="review-card">
            <div>
                <div class="review-stars">&#9733;&#9733;&#9733;&#9733;&#9733;</div>
                <p>{review['quote']}</p>
            </div>
            <div class="reviewer">{review['author']}</div>
        </div>
        """
        for review in review_cards
    )

    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-grid">
                <div>
                    <div class="hero-kicker">Welcome to GreenVest</div>
                    <div class="hero-brand">
                        <img class="hero-logo" src="{LOGO_DATA_URI}" alt="GreenVest logo">
                        <div>
                            <h1 class="hero-title">GreenVest</h1>
                            <p class="hero-copy">
                                Sustainable investing should feel clear, modern, and reassuring.
                                GreenVest helps investors compare return, risk, and ESG quality without
                                forcing them through a cluttered workflow.
                            </p>
                        </div>
                    </div>
                    <div class="hero-actions">
                        <a class="hero-button green" href="?launch=manual">Create Your Own</a>
                        <a class="hero-button blue" href="?launch=guided">Generate For Me</a>
                    </div>
                    <div class="hero-note">
                        Use the green path for a self-directed portfolio build, or the blue path to generate
                        a portfolio from investor preferences.
                    </div>
                </div>
                <div class="hero-panel">
                    <div class="hero-panel-copy">
                        <h4>What changes in this redesign</h4>
                        <div class="hero-points">
                            <div class="hero-point">
                                <strong>Cleaner first impression</strong>
                                A branded welcome screen replaces the old popup-driven entry flow.
                            </div>
                            <div class="hero-point">
                                <strong>Better investor readability</strong>
                                The dashboard moves from tabs into a guided build-and-insight layout.
                            </div>
                            <div class="hero-point">
                                <strong>Sharper visuals</strong>
                                Only the ESG frontier and future value charts remain, both enlarged and simplified.
                            </div>
                        </div>
                    </div>
                    <img class="hero-mascot" src="{GLASSES_DATA_URI}" alt="GreenVest mascot">
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="fact-grid">
            <div class="fact-card">
                <div class="section-kicker">What it is</div>
                <h4>Sustainable investing in plain language</h4>
                <p>
                    Sustainable investing looks at financial return alongside environmental, social,
                    and governance performance. Instead of asking only "what could this earn?",
                    it also asks "how is this business being run, and what risks or opportunities does that create?"
                </p>
            </div>
            <div class="fact-card">
                <div class="section-kicker">Why it matters</div>
                <h4>Why investors choose this approach</h4>
                <p>
                    Investors often use sustainable strategies to align with their values, manage long-run
                    regulatory and reputational risk, and back companies that may be better positioned for
                    future policy, consumer, and market shifts.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <section class="review-shell">
            <div class="review-intro">
                <div>
                    <div class="section-kicker">Investor reactions</div>
                    <h3>Scrolling social proof for the welcome page</h3>
                </div>
                <div class="hero-note">Reviews drift left to right and fade softly at the edges.</div>
            </div>
            <div class="review-rail">
                <div class="review-track">
                    {review_html}
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    defaults = {
        "entered_app": False,
        "show_profile_builder": False,
        "setup_mode": None,
        "loader_complete": False,
        "onboarding_done": False,
        "onboarding_step": 1,
        "q1": 3,
        "q2": 3,
        "q3": 3,
        "goal": 2,
        "e_w": 3,
        "s_w": 3,
        "g_w": 3,
        "excl_tobacco": False,
        "excl_weapons": False,
        "excl_gambling": False,
        "excl_fossil": False,
        "excl_alcohol": False,
        "gamma": 4.0,
        "lambda_esg": 0.06,
        "gamma_val": 4.0,
        "lambda_val": 0.06,
        "profile": "Balanced",
        "goal_label": "Long-term growth",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def handle_entry_actions():
    action = get_query_param("launch")
    if action == "manual":
        st.session_state.entered_app = True
        st.session_state.onboarding_done = True
        st.session_state.setup_mode = "manual"
        st.session_state.profile = "Self-directed"
        st.session_state.goal_label = "Custom portfolio"
        st.session_state.show_profile_builder = False
        clear_query_params()
        st.rerun()

    if action == "guided":
        st.session_state.show_profile_builder = True
        st.session_state.setup_mode = "guided"
        st.session_state.onboarding_step = 1
        clear_query_params()
        st.rerun()


def render_dashboard():
    gamma_used = st.session_state.gamma_val
    lambda_used = st.session_state.lambda_val
    e_w = st.session_state.e_w
    s_w = st.session_state.s_w
    g_w = st.session_state.g_w

    st.markdown(
        f"""
        <section class="dashboard-hero">
            <div class="dashboard-head">
                <div>
                    <div class="section-kicker">Investor workspace</div>
                    <h2 class="dashboard-title">Build a clearer sustainable portfolio</h2>
                    <p class="dashboard-copy">
                        The dashboard now flows from portfolio inputs to recommendation to visuals in a single pass,
                        so investors can follow the story without bouncing between tabs.
                    </p>
                    <div class="chip-row">
                        <span class="chip">{st.session_state.profile}</span>
                        <span class="chip blue">Goal: {st.session_state.goal_label}</span>
                        <span class="chip">gamma = {gamma_used}</span>
                        <span class="chip">lambda = {lambda_used}</span>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    top_actions = st.columns([1.35, 1, 1.05])
    with top_actions[2]:
        if st.button("Update Preferences"):
            st.session_state.show_profile_builder = True
            st.session_state.onboarding_step = 1
            st.rerun()

    if st.session_state.show_profile_builder:
        render_profile_builder(editing=True)
        st.write("")

    builder_col, insight_col = st.columns([1.08, 0.92], gap="large")

    with builder_col:
        st.markdown(
            """
            <div class="section-kicker">Portfolio builder</div>
            <h3 class="section-title">Shape the assets and market assumptions</h3>
            <p class="section-copy">
                Change the asset characteristics, ESG pillar scores, and market context.
                The recommendation refreshes immediately in the panel beside it.
            </p>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Advanced controls for gamma and lambda", expanded=False):
            st.caption(
                "Use these only if you want to override the profile-derived risk aversion and ESG preference values."
            )

            gamma_cols = st.columns([4, 1])
            with gamma_cols[0]:
                st.slider(
                    "Risk aversion",
                    1.0,
                    10.0,
                    st.session_state.gamma_val,
                    0.1,
                    key="_g_sl",
                    on_change=sync_gamma_slider,
                    help="Higher gamma means the investor is more sensitive to portfolio risk.",
                )
            with gamma_cols[1]:
                st.number_input(
                    "gamma exact",
                    1.0,
                    10.0,
                    st.session_state.gamma_val,
                    0.1,
                    key="_g_ni",
                    on_change=sync_gamma_input,
                    label_visibility="collapsed",
                )

            lambda_cols = st.columns([4, 1])
            with lambda_cols[0]:
                st.slider(
                    "ESG preference",
                    0.01,
                    0.20,
                    st.session_state.lambda_val,
                    0.005,
                    key="_l_sl",
                    on_change=sync_lambda_slider,
                    help="Higher lambda gives sustainability a stronger role in the utility score.",
                )
            with lambda_cols[1]:
                st.number_input(
                    "lambda exact",
                    0.01,
                    0.20,
                    st.session_state.lambda_val,
                    0.005,
                    key="_l_ni",
                    on_change=sync_lambda_input,
                    format="%.3f",
                    label_visibility="collapsed",
                )

            st.info(
                f"Active values -> gamma = {st.session_state.gamma_val:.1f} | "
                f"lambda = {st.session_state.lambda_val:.3f}"
            )

        st.write("")
        market_cols = st.columns(2)
        with market_cols[0]:
            rf = st.number_input(
                "Risk-free rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=4.5,
                step=0.1,
            ) / 100
        with market_cols[1]:
            invest = st.number_input(
                "Investment amount (GBP)",
                min_value=100.0,
                max_value=1_000_000.0,
                value=10_000.0,
                step=500.0,
            )

        st.markdown(
            f"""
            <div class="info-note">
                Excluded sectors: <strong>{", ".join(excluded_sector_labels())}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        excluded_sectors = get_excluded_sectors()
        assets = {}

        for i in [1, 2]:
            st.write("")
            st.markdown(
                f"""
                <div class="section-kicker">Asset {i}</div>
                <h4 style="margin:0.2rem 0 0.35rem;">Asset {i} inputs</h4>
                """,
                unsafe_allow_html=True,
            )
            cols = st.columns([1.05, 1, 0.8])

            with cols[0]:
                name = st.text_input("Name", value=f"Asset {i}", key=f"name{i}")
                sector = st.selectbox(
                    "Sector",
                    SECTORS,
                    index=0 if i == 1 else 1,
                    key=f"sector{i}",
                )
                mu = st.number_input(
                    "Expected return (%)",
                    -100.0,
                    500.0,
                    8.0 if i == 1 else 5.0,
                    0.1,
                    key=f"mu{i}",
                ) / 100
                sigma = st.number_input(
                    "Std deviation (%)",
                    0.01,
                    500.0,
                    15.0 if i == 1 else 10.0,
                    0.1,
                    key=f"sigma{i}",
                ) / 100

            with cols[1]:
                st.caption("ESG pillar scores")
                e_score = st.slider("Environmental", 0.0, 100.0, 70.0 if i == 1 else 50.0, key=f"e{i}")
                s_score = st.slider("Social", 0.0, 100.0, 65.0 if i == 1 else 55.0, key=f"s{i}")
                g_score = st.slider("Governance", 0.0, 100.0, 60.0 if i == 1 else 45.0, key=f"g{i}")

            with cols[2]:
                esg_last = st.number_input(
                    "Last year ESG",
                    0.0,
                    100.0,
                    65.0 if i == 1 else 48.0,
                    key=f"esg_last{i}",
                )

            esg_composite = composite_esg(e_score, s_score, g_score, e_w, s_w, g_w) / 100
            momentum = esg_momentum(composite_esg(e_score, s_score, g_score, e_w, s_w, g_w), esg_last)
            adjusted_esg = min(esg_composite + 0.05 * momentum, 1.0)
            is_excluded = sector in excluded_sectors

            st.caption(
                f"Composite ESG: {esg_composite * 100:.1f} [{esg_rating(esg_composite)}] | "
                f"Momentum: {momentum * 100:+.1f}% | Adjusted ESG: {adjusted_esg * 100:.1f}"
            )

            if esg_composite >= 0.60 and g_score / 100 < 0.35:
                st.warning(f"Greenwashing alert: {name} has a strong overall ESG score but weak governance.")

            if is_excluded:
                st.error(
                    f"{name} sits inside an excluded sector ({sector}), so GreenVest will force its allocation to 0%."
                )

            assets[i] = {
                "name": name,
                "sector": sector,
                "mu": mu,
                "sigma": sigma,
                "e": e_score,
                "s": s_score,
                "g": g_score,
                "esg_last": esg_last,
                "esg_c": esg_composite,
                "mom": momentum,
                "esg_a": adjusted_esg,
                "is_excluded": is_excluded,
            }

        st.write("")
        rho = st.slider(
            "Correlation between the two assets",
            -1.0,
            1.0,
            0.3,
            0.01,
            help="-1 means the assets move opposite to each other. 1 means they move together.",
        )

    a1 = assets[1]
    a2 = assets[2]
    both_excluded = a1["is_excluded"] and a2["is_excluded"]
    force_w1 = None
    if a1["is_excluded"] and not a2["is_excluded"]:
        force_w1 = 0.0
    elif a2["is_excluded"] and not a1["is_excluded"]:
        force_w1 = 1.0

    results = {}
    summary_pdf_bytes = None
    if not both_excluded:
        (
            weights,
            mu_grid,
            sigma_grid,
            esg_grid,
            utility_grid,
            esg_adjusted_grid,
            idx,
        ) = optimise(
            a1["mu"],
            a2["mu"],
            a1["sigma"],
            a2["sigma"],
            a1["esg_a"],
            a2["esg_a"],
            rho,
            gamma_used,
            lambda_used,
            force_w1=force_w1,
        )

        w1 = float(weights[idx])
        w2 = 1 - w1
        mu_opt = float(mu_grid[idx])
        sigma_opt = float(sigma_grid[idx])
        esg_opt = float(esg_grid[idx])
        utility_opt = float(utility_grid[idx])
        sharpe_opt = (mu_opt - rf) / sigma_opt if sigma_opt > 0 else float("nan")
        esg_sharpe_opt = esg_sharpe(mu_opt, rf, sigma_opt, lambda_used, esg_opt)
        impact_score = round(esg_opt * 100, 1)
        tradeoff = lambda_used * esg_opt - gamma_used * sigma_opt

        benchmark_return = None
        ret_cost = 0.0
        idx_financial = None
        idx_low_risk = None
        mu_50 = None

        if force_w1 is None:
            _, mu_ms, sigma_ms, esg_ms, _, esg_adjusted_ms, idx_financial = optimise(
                a1["mu"],
                a2["mu"],
                a1["sigma"],
                a2["sigma"],
                a1["esg_a"],
                a2["esg_a"],
                rho,
                gamma_used,
                0.0,
            )
            idx_low_risk = int(np.argmin(sigma_grid))
            benchmark_return = p_return(0.5, a1["mu"], a2["mu"])
            mu_50 = benchmark_return
            ret_cost = mu_ms[idx_financial] * 100 - mu_opt * 100
        else:
            mu_ms = sigma_ms = esg_adjusted_ms = None

        results = {
            "weights": weights,
            "mu_grid": mu_grid,
            "sigma_grid": sigma_grid,
            "esg_grid": esg_grid,
            "esg_adjusted_grid": esg_adjusted_grid,
            "idx": idx,
            "w1": w1,
            "w2": w2,
            "mu_opt": mu_opt,
            "sigma_opt": sigma_opt,
            "esg_opt": esg_opt,
            "utility_opt": utility_opt,
            "sharpe_opt": sharpe_opt,
            "esg_sharpe_opt": esg_sharpe_opt,
            "impact_score": impact_score,
            "tradeoff": tradeoff,
            "benchmark_return": benchmark_return,
            "ret_cost": ret_cost,
            "idx_financial": idx_financial,
            "idx_low_risk": idx_low_risk,
            "mu_ms": mu_ms,
            "sigma_ms": sigma_ms,
            "esg_adjusted_ms": esg_adjusted_ms,
            "mu_50": mu_50,
        }
        summary_pdf_bytes = build_summary_pdf_bytes(
            a1,
            a2,
            results,
            invest,
            st.session_state.profile,
            st.session_state.goal_label,
            gamma_used,
            lambda_used,
            e_w,
            s_w,
            g_w,
            ", ".join(excluded_sector_labels()),
        )

    with insight_col:
        st.markdown(
            """
            <div class="section-kicker">GreenVest recommendation</div>
            <h3 class="section-title">Read the portfolio as a story, not a spreadsheet</h3>
            <p class="section-copy">
                The panel below summarises the recommended mix, why it fits the investor, and how
                the ESG profile compares across the two assets.
            </p>
            """,
            unsafe_allow_html=True,
        )

        excluded_text = ", ".join(excluded_sector_labels())
        st.markdown(
            f"""
            <div class="spotlight-card">
                <div class="section-kicker">Investor profile</div>
                <h3>{st.session_state.profile} outlook with {st.session_state.goal_label.lower()} in mind</h3>
                <p>
                    Current profile inputs set gamma to <strong>{gamma_used:.1f}</strong> and lambda to
                    <strong>{lambda_used:.3f}</strong>. Environmental, social, and governance priorities are
                    weighted <strong>{e_w}/5</strong>, <strong>{s_w}/5</strong>, and <strong>{g_w}/5</strong>.
                    Excluded sectors: <strong>{excluded_text}</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("How GreenVest scores ESG", expanded=False):
            st.markdown(
                """
                1. GreenVest starts with the environmental, social, and governance sub-scores you enter for each asset.
                2. Those pillar scores are combined into one composite ESG score using the investor's own E, S, and G priority weights.
                3. A modest momentum adjustment is applied by comparing today's composite ESG score with last year's ESG level, which gives improving companies some credit without overpowering the model.
                4. The final portfolio recommendation is chosen by the utility formula `U = mu - (gamma / 2) * sigma^2 + lambda * ESG`, so return, risk, and sustainability all matter at the same time.
                5. GreenVest also enforces sector exclusions and raises a greenwashing warning when the overall ESG score looks strong but governance is weak.
                """
            )

        if both_excluded:
            st.error(
                "Both assets sit inside excluded sectors. Update the exclusions or change the asset sectors to restore a valid portfolio."
            )
        else:
            high_esg_asset = a1["name"] if a1["esg_a"] >= a2["esg_a"] else a2["name"]
            high_return_asset = a1["name"] if a1["mu"] >= a2["mu"] else a2["name"]
            gamma_desc = (
                "high risk aversion" if gamma_used >= 7 else "moderate risk aversion" if gamma_used >= 4 else "lower risk aversion"
            )
            lambda_desc = (
                "strong sustainability preference"
                if lambda_used >= 0.12
                else "balanced sustainability preference"
                if lambda_used >= 0.06
                else "lighter sustainability preference"
            )

            st.markdown(
                f"""
                <div class="spotlight-card">
                    <div class="section-kicker">Recommended mix</div>
                    <h3>{a1["name"]}: {results["w1"] * 100:.1f}% | {a2["name"]}: {results["w2"] * 100:.1f}%</h3>
                    <p>
                        GreenVest leans toward <strong>{high_esg_asset}</strong> for sustainability strength while
                        preserving return support from <strong>{high_return_asset}</strong>. With
                        <strong>{gamma_desc}</strong> and a <strong>{lambda_desc}</strong>, the optimiser is searching
                        for the mix that respects both financial and ESG preferences without cluttering the story.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metric_row_1 = st.columns(2)
            metric_row_1[0].metric("Expected return", f"{results['mu_opt'] * 100:.2f}%")
            metric_row_1[1].metric("Portfolio risk", f"{results['sigma_opt'] * 100:.2f}%")

            metric_row_2 = st.columns(2)
            metric_row_2[0].metric("Sharpe ratio", f"{results['sharpe_opt']:.3f}")
            metric_row_2[1].metric("ESG score", f"{results['esg_opt'] * 100:.1f} / 100")

            metric_row_3 = st.columns(2)
            metric_row_3[0].metric("ESG-adjusted Sharpe", f"{results['esg_sharpe_opt']:.3f}")
            metric_row_3[1].metric("Impact score", f"{results['impact_score']:.1f}")

            metric_row_4 = st.columns(2)
            metric_row_4[0].metric("Trade-off score", f"{results['tradeoff']:.4f}")
            metric_row_4[1].metric("Utility value", f"{results['utility_opt']:.5f}")

            if force_w1 is not None:
                excluded_name = a1["name"] if a1["is_excluded"] else a2["name"]
                remaining_name = a2["name"] if a1["is_excluded"] else a1["name"]
                st.warning(
                    f"Hard exclusion applied: {excluded_name} is blocked by the investor rules, so the portfolio moves fully into {remaining_name}."
                )
            elif results["ret_cost"] > 0.5:
                st.info(
                    f"Compared with a purely financial benchmark, this sustainable tilt gives up about {results['ret_cost']:.2f}% of expected return."
                )
            else:
                st.success("The current sustainable preference set does not create a meaningful return penalty in this two-asset setup.")

            esg_df = pd.DataFrame(
                {
                    "Pillar": ["Environmental", "Social", "Governance", "Composite", "Adjusted"],
                    a1["name"]: [a1["e"], a1["s"], a1["g"], f"{a1['esg_c'] * 100:.1f}", f"{a1['esg_a'] * 100:.1f}"],
                    a2["name"]: [a2["e"], a2["s"], a2["g"], f"{a2['esg_c'] * 100:.1f}", f"{a2['esg_a'] * 100:.1f}"],
                    "Investor weight": [f"{e_w}/5", f"{s_w}/5", f"{g_w}/5", "-", "-"],
                }
            )

            st.write("")
            st.markdown(
                """
                <div class="section-kicker">ESG detail</div>
                <h4 style="margin:0.2rem 0 0.6rem;">Pillar-level comparison</h4>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(esg_df, use_container_width=True, hide_index=True)

            checkpoint_years = [5, 10, 20, 30]
            checkpoint_df = pd.DataFrame(
                {
                    "Horizon": [f"{year} years" for year in checkpoint_years],
                    "GreenVest": [f"GBP {future_value(invest, results['mu_opt'], year):,.0f}" for year in checkpoint_years],
                }
            )
            if results["benchmark_return"] is not None:
                checkpoint_df["50 / 50"] = [
                    f"GBP {future_value(invest, results['benchmark_return'], year):,.0f}" for year in checkpoint_years
                ]

            with st.expander("Projection checkpoints", expanded=False):
                st.dataframe(checkpoint_df, use_container_width=True, hide_index=True)

            with st.expander("Export for meetings and submissions", expanded=False):
                st.write("Download a one-page PDF summary of the current recommendation.")
                st.download_button(
                    "Download one-page PDF summary",
                    data=summary_pdf_bytes,
                    file_name="greenvest-portfolio-summary.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    st.write("")
    st.markdown(
        """
        <div class="section-kicker">Visual story</div>
        <h3 class="section-title">Two clearer charts for investors</h3>
        <p class="section-copy">
            The chart area now focuses on the two visuals that matter most: where the efficient sustainable portfolio sits,
            and how the chosen strategy compounds over time.
        </p>
        """,
        unsafe_allow_html=True,
    )

    chart_cols = st.columns(2, gap="large")
    if both_excluded:
        with chart_cols[0]:
            st.error("Frontier chart unavailable until at least one asset falls outside the excluded sectors.")
        with chart_cols[1]:
            st.info("Future value projection will appear again once the portfolio becomes investable.")
        return

    if force_w1 is None:
        frontier_fig = build_frontier_chart(
            results["sigma_grid"],
            results["esg_adjusted_grid"],
            results["esg_grid"],
            results["sigma_opt"],
            results["esg_adjusted_grid"][results["idx"]],
            results["sigma_ms"][results["idx_financial"]],
            results["esg_adjusted_ms"][results["idx_financial"]],
            results["sigma_grid"][results["idx_low_risk"]],
            results["esg_adjusted_grid"][results["idx_low_risk"]],
        )
        with chart_cols[0]:
            st.markdown("#### ESG efficient frontier")
            st.pyplot(frontier_fig)
            plt.close(frontier_fig)
            st.markdown(
                '<div class="chart-caption">Legend markers and colors have been reduced so the data itself stays in focus.</div>',
                unsafe_allow_html=True,
            )
    else:
        with chart_cols[0]:
            st.markdown("#### ESG efficient frontier")
            st.info(
                "The ESG efficient frontier only appears when both assets remain investable. Remove the exclusion to compare mixes again."
            )

    selected_label = a1["name"] if results["w1"] >= results["w2"] else a2["name"]
    future_fig = build_future_value_chart(
        invest,
        results["mu_opt"],
        benchmark_return=results["benchmark_return"],
        selected_label=selected_label,
    )
    with chart_cols[1]:
        st.markdown("#### Future value projection")
        st.pyplot(future_fig)
        plt.close(future_fig)
        st.markdown(
            '<div class="chart-caption">Projected values use the expected return inputs you set above, so investors can test scenarios instantly.</div>',
            unsafe_allow_html=True,
        )


initialize_session_state()
inject_styles()
handle_entry_actions()

if not st.session_state.loader_complete:
    st.session_state.loader_complete = True
    render_loader()

if not st.session_state.entered_app:
    render_landing_page()
    if st.session_state.show_profile_builder:
        st.write("")
        render_profile_builder(editing=False)
    st.stop()

render_dashboard()
