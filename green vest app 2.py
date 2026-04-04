import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GreenVest | ESG Portfolio Optimiser",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f4faf5; color: #1a2e1e; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #e4f0e8; border-radius: 8px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px; padding: 8px 28px;
        font-weight: 600; color: #2d6a4f; background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #2d6a4f !important; color: white !important;
    }

    div[data-testid="metric-container"] {
        background: white; border: 1px solid #c3dfc9;
        border-radius: 10px; padding: 14px 18px;
        border-left: 4px solid #2d6a4f;
    }

    .stButton > button {
        background-color: #2d6a4f; color: white; border: none;
        border-radius: 6px; padding: 10px 28px; font-weight: 600; width: 100%;
    }
    .stButton > button:hover {
        background-color: #1e5239 !important; color: white !important;
    }

    hr { border-color: #c3dfc9; margin: 20px 0; }
    h1, h2, h3, h4 { color: #1e5239; }

    /* Welcome screen */
    .welcome-wrap {
        display: flex; flex-direction: column;
        align-items: center; justify-content: center; min-height: 70vh;
    }
    .welcome-card {
        background: white; border: 1px solid #c3dfc9; border-radius: 18px;
        padding: 56px 64px; text-align: center;
        box-shadow: 0 6px 32px rgba(45,106,79,0.10);
        max-width: 580px; width: 100%;
    }
    .welcome-logo {
        font-size: 2.6rem; font-weight: 800; color: #1e5239;
        letter-spacing: -0.5px; margin-bottom: 6px;
    }
    .welcome-tagline {
        font-size: 0.88rem; font-weight: 600; color: #2d6a4f;
        letter-spacing: 1.8px; text-transform: uppercase; margin-bottom: 20px;
    }
    .welcome-body {
        font-size: 1.02rem; color: #3d5e47; line-height: 1.7; margin-bottom: 32px;
    }
    .welcome-pills {
        display: flex; gap: 10px; justify-content: center;
        margin-bottom: 36px; flex-wrap: wrap;
    }
    .pill {
        background: #e4f0e8; color: #2d6a4f; border-radius: 20px;
        padding: 5px 16px; font-size: 0.82rem; font-weight: 600;
    }

    /* Summary card */
    .summary-card {
        background: white; border: 1px solid #c3dfc9;
        border-top: 4px solid #2d6a4f; border-radius: 12px;
        padding: 24px 32px; margin-bottom: 28px;
    }
    .summary-title { font-size: 1rem; font-weight: 700; color: #1e5239; margin-bottom: 14px; }
    .summary-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    .summary-table td { padding: 5px 0; }
    .summary-label { color: #4a7c5e; width: 160px; }
    .summary-value { font-weight: 600; color: #1a2e1e; padding-right: 32px; }

    /* Insight / explanation box */
    .insight-box {
        background: #edf7f0; border-left: 4px solid #2d6a4f;
        border-radius: 0 8px 8px 0; padding: 16px 20px; margin: 16px 0;
        font-size: 0.93rem; color: #1e5239; line-height: 1.65;
    }
    .insight-box strong { color: #1e5239; }

    /* Chart section label */
    .chart-section {
        font-size: 0.8rem; font-weight: 700; color: #2d6a4f;
        text-transform: uppercase; letter-spacing: 1.2px;
        margin: 20px 0 6px 0; padding-bottom: 6px;
        border-bottom: 1px solid #c3dfc9;
    }

    /* Step label in dialog */
    .step-label {
        font-size: 0.78rem; color: #4a7c5e; text-align: right;
        margin-bottom: 6px; font-weight: 500;
        letter-spacing: 0.5px; text-transform: uppercase;
    }

    /* Override panel */
    .override-label {
        font-size: 0.78rem; color: #4a7c5e; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────
SECTORS = [
    "Clean Energy", "Technology", "Healthcare", "Financial Services",
    "Consumer Goods", "Industrials", "Real Estate",
    "Tobacco", "Weapons / Defence", "Gambling", "Fossil Fuels / Energy", "Alcohol",
]
EXCLUDED_SECTOR_MAP = {
    "Tobacco":             "Tobacco",
    "Weapons / Defence":   "Weapons",
    "Gambling":            "Gambling",
    "Fossil Fuels / Energy": "Fossil Fuels",
    "Alcohol":             "Alcohol",
}


# ── Pure functions ─────────────────────────────────────────────────────────────
def p_return(w1, mu1, mu2):
    return w1 * mu1 + (1 - w1) * mu2

def p_std(w1, s1, s2, rho):
    return np.sqrt(w1**2*s1**2 + (1-w1)**2*s2**2 + 2*rho*w1*(1-w1)*s1*s2)

def p_esg(w1, e1, e2):
    return w1 * e1 + (1 - w1) * e2

def p_util(mu, sig, esg, g, l):
    return mu - (g / 2) * sig**2 + l * esg

def composite_esg(e, s, g, we, ws, wg):
    t = we + ws + wg
    return (we * e + ws * s + wg * g) / t if t > 0 else 0

def esg_momentum(now, last):
    return (now - last) / last if last > 0 else 0

def esg_rating(score):
    for thresh, rating in [(0.85,"AAA"),(0.70,"AA"),(0.55,"A"),
                           (0.40,"BBB"),(0.25,"BB"),(0.10,"B")]:
        if score >= thresh:
            return rating
    return "CCC"

def esg_sharpe(mu, rf, sig, lam, esg):
    return (mu - rf + lam * esg) / sig if sig > 0 else float("nan")

def future_value(pv, r, years):
    return pv * (1 + r) ** years

def optimise(mu1, mu2, s1, s2, e1, e2, rho, gamma, lam, n=2000,
             force_w1=None):
    """
    force_w1: if not None, weight of asset 1 is fixed (hard sector exclusion).
    """
    if force_w1 is not None:
        w   = np.array([force_w1])
        mu_g = p_return(w, mu1, mu2)
        sg_g = p_std(w, s1, s2, rho)
        eg_g = p_esg(w, e1, e2)
        ut_g = p_util(mu_g, sg_g, eg_g, gamma, lam)
        return w, mu_g, sg_g, eg_g, ut_g, mu_g + lam * eg_g, 0

    w    = np.linspace(0, 1, n)
    mu_g = p_return(w, mu1, mu2)
    sg_g = p_std(w, s1, s2, rho)
    eg_g = p_esg(w, e1, e2)
    ut_g = p_util(mu_g, sg_g, eg_g, gamma, lam)
    idx  = np.argmax(ut_g)
    return w, mu_g, sg_g, eg_g, ut_g, mu_g + lam * eg_g, idx

def derive_lambda(e_w, s_w, g_w, excl_count, goal_lam):
    """
    Rescaled to 0.01–0.15 to keep ESG influence economically proportionate.
    At lambda=0.10 with ESG=0.65: adds ~0.065 utility vs a return of ~0.08.
    """
    base = ((e_w + s_w + g_w) / 3 / 5) * 0.11 + 0.01
    lam  = min(base + 0.005 * excl_count + goal_lam, 0.20)
    return round(lam, 3)


# ── Session state defaults ─────────────────────────────────────────────────────
_defaults = {
    'onboarding_done': False,
    'onboarding_step': 1,
    'show_dialog':     False,
    'is_update':       False,   # True when re-opening to edit (enables cancel)
    'q1': 3, 'q2': 3, 'q3': 3,
    'goal': 2,
    'e_w': 3, 's_w': 3, 'g_w': 3,
    'excl_tobacco': False, 'excl_weapons': False, 'excl_gambling': False,
    'excl_fossil':  False, 'excl_alcohol': False,
    'gamma':      4.0,
    'lambda_esg': 0.06,
    'gamma_val':  4.0,
    'lambda_val': 0.06,
    'profile':    'Balanced',
    'goal_label': 'Long-term growth',
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sync helpers for slider + number_input ─────────────────────────────────────
def _sync_gamma_slider():
    st.session_state.gamma_val = st.session_state._g_sl

def _sync_gamma_input():
    st.session_state.gamma_val = st.session_state._g_ni

def _sync_lam_slider():
    st.session_state.lambda_val = st.session_state._l_sl

def _sync_lam_input():
    st.session_state.lambda_val = st.session_state._l_ni


# ── Multi-step onboarding dialog ───────────────────────────────────────────────
@st.dialog("Your Investment Profile", width="large")
def onboarding_dialog():
    TOTAL = 5
    step  = st.session_state.onboarding_step
    is_update = st.session_state.is_update

    st.markdown(f'<div class="step-label">Step {step} of {TOTAL}</div>',
                unsafe_allow_html=True)
    st.progress(step / TOTAL)
    st.write("")

    # ── Step 1 — Reaction to loss ─────────────────────────────────────────────
    if step == 1:
        st.markdown("#### If your portfolio dropped 25%, what would you do?")
        st.caption("This helps us understand how you handle market volatility. "
                   "There is no wrong answer.")
        st.write("")
        choice = st.radio(" ", options=[1,2,3,4], format_func=lambda x: {
            1: "Sell everything — I cannot afford to lose more",
            2: "Reduce exposure — I want to limit further losses",
            3: "Hold steady — I trust the market will recover",
            4: "Buy more — downturns are buying opportunities",
        }[x], index=st.session_state.q1 - 1, label_visibility="collapsed")
        st.write("")
        c_cancel, _, c_next = st.columns([1, 2, 1])
        if is_update:
            with c_cancel:
                if st.button("Cancel", key="cancel1"):
                    st.session_state.show_dialog     = False
                    st.session_state.onboarding_step = 1
                    st.rerun()
        with c_next:
            if st.button("Next", key="next1"):
                st.session_state.q1 = choice
                st.session_state.onboarding_step = 2
                st.rerun()

    # ── Step 2 — Return objective ─────────────────────────────────────────────
    elif step == 2:
        st.markdown("#### What is your primary investment objective?")
        st.caption("This shapes how aggressively we build your portfolio.")
        st.write("")
        choice = st.radio(" ", options=[1,2,3,4], format_func=lambda x: {
            1: "Preserve capital — protect what I have",
            2: "Generate income — steady, reliable returns",
            3: "Long-term growth — grow my wealth steadily",
            4: "Maximise growth — highest possible returns",
        }[x], index=st.session_state.q2 - 1, label_visibility="collapsed")
        st.write("")
        c_back, _, c_next = st.columns([1, 2, 1])
        with c_back:
            if st.button("Back", key="back2"):
                st.session_state.q2 = choice
                st.session_state.onboarding_step = 1
                st.rerun()
        with c_next:
            if st.button("Next", key="next2"):
                st.session_state.q2 = choice
                st.session_state.onboarding_step = 3
                st.rerun()

    # ── Step 3 — Time horizon ─────────────────────────────────────────────────
    elif step == 3:
        st.markdown("#### How long do you plan to stay invested?")
        st.caption("A longer horizon allows for more risk and higher potential reward.")
        st.write("")
        choice = st.radio(" ", options=[1,2,3,4], format_func=lambda x: {
            1: "Less than 2 years",
            2: "2 to 5 years",
            3: "5 to 10 years",
            4: "10 years or more",
        }[x], index=st.session_state.q3 - 1, label_visibility="collapsed")
        st.write("")
        c_back, _, c_next = st.columns([1, 2, 1])
        with c_back:
            if st.button("Back", key="back3"):
                st.session_state.q3 = choice
                st.session_state.onboarding_step = 2
                st.rerun()
        with c_next:
            if st.button("Next", key="next3"):
                st.session_state.q3 = choice
                st.session_state.onboarding_step = 4
                st.rerun()

    # ── Step 4 — Investment goal ──────────────────────────────────────────────
    elif step == 4:
        st.markdown("#### What are you investing for?")
        st.caption("Your goal influences how we balance return, risk, and ESG.")
        st.write("")
        choice = st.radio(" ", options=[1,2,3,4], format_func=lambda x: {
            1: "Retirement — build long-term financial security",
            2: "Long-term growth — grow wealth over time",
            3: "Ethical investing — prioritise ESG alongside returns",
            4: "Short-term profit — maximise near-term gains",
        }[x], index=st.session_state.goal - 1, label_visibility="collapsed")
        st.write("")
        c_back, _, c_next = st.columns([1, 2, 1])
        with c_back:
            if st.button("Back", key="back4"):
                st.session_state.goal = choice
                st.session_state.onboarding_step = 3
                st.rerun()
        with c_next:
            if st.button("Next", key="next4"):
                st.session_state.goal = choice
                st.session_state.onboarding_step = 5
                st.rerun()

    # ── Step 5 — ESG priorities ───────────────────────────────────────────────
    elif step == 5:
        st.markdown("#### How much do you value ESG factors?")
        st.caption("Rate each pillar 0 (not important) to 5 (essential). "
                   "These weights personalise your ESG score.")
        st.write("")
        e_w = st.slider("Environmental", 0, 5, st.session_state.e_w,
                        help="Climate impact, emissions, resource use")
        s_w = st.slider("Social",        0, 5, st.session_state.s_w,
                        help="Labour rights, community, diversity")
        g_w = st.slider("Governance",    0, 5, st.session_state.g_w,
                        help="Board structure, transparency, ethics")
        st.write("")
        st.caption("Exclude any sectors from your portfolio?")
        c1,c2,c3,c4,c5 = st.columns(5)
        excl_tobacco  = c1.checkbox("Tobacco",      value=st.session_state.excl_tobacco)
        excl_weapons  = c2.checkbox("Weapons",      value=st.session_state.excl_weapons)
        excl_gambling = c3.checkbox("Gambling",     value=st.session_state.excl_gambling)
        excl_fossil   = c4.checkbox("Fossil Fuels", value=st.session_state.excl_fossil)
        excl_alcohol  = c5.checkbox("Alcohol",      value=st.session_state.excl_alcohol)
        st.write("")
        c_back, _, c_done = st.columns([1, 2, 1])
        with c_back:
            if st.button("Back", key="back5"):
                st.session_state.e_w = e_w; st.session_state.s_w = s_w
                st.session_state.g_w = g_w
                st.session_state.onboarding_step = 4
                st.rerun()
        with c_done:
            if st.button("Complete Profile", key="done5"):
                st.session_state.e_w=e_w; st.session_state.s_w=s_w
                st.session_state.g_w=g_w
                st.session_state.excl_tobacco=excl_tobacco
                st.session_state.excl_weapons=excl_weapons
                st.session_state.excl_gambling=excl_gambling
                st.session_state.excl_fossil=excl_fossil
                st.session_state.excl_alcohol=excl_alcohol

                q1,q2,q3 = st.session_state.q1,st.session_state.q2,st.session_state.q3
                score      = ((5-q1)+(5-q2)+q3)/3
                gamma_base = round(2+(score-1)*(8/3),1)
                profile    = ("Conservative" if gamma_base>=7
                              else "Balanced" if gamma_base>=4 else "Aggressive")
                goal       = st.session_state.goal
                gamma      = round(max(1.0, gamma_base+{1:+2,2:0,3:0,4:-1}[goal]),1)
                goal_lam   = {1:0,2:0,3:0.02,4:0}[goal]
                excl_count = sum([excl_tobacco,excl_weapons,
                                  excl_gambling,excl_fossil,excl_alcohol])
                lambda_esg = derive_lambda(e_w,s_w,g_w,excl_count,goal_lam)

                _goal_labels = {1:"Retirement",2:"Long-term growth",
                                3:"Ethical investing",4:"Short-term profit"}
                st.session_state.gamma      = gamma
                st.session_state.gamma_val  = gamma
                st.session_state.lambda_esg = lambda_esg
                st.session_state.lambda_val = lambda_esg
                st.session_state.profile    = profile
                st.session_state.goal_label = _goal_labels[goal]
                st.session_state.onboarding_done = True
                st.session_state.is_update       = False
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.onboarding_done:
    st.markdown("""
    <div class="welcome-wrap">
      <div class="welcome-card">
        <div class="welcome-logo">GreenVest</div>
        <div class="welcome-tagline">ESG Portfolio Optimiser</div>
        <div class="welcome-body">
          Build a portfolio that balances financial performance with
          environmental, social, and governance values.<br><br>
          Answer a few short questions and we will find your optimal allocation.
        </div>
        <div class="welcome-pills">
          <span class="pill">Risk profiling</span>
          <span class="pill">ESG scoring</span>
          <span class="pill">Efficient frontier</span>
          <span class="pill">Portfolio optimisation</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([2,1,2])
    with col:
        if st.button("Get Started"):
            st.session_state.show_dialog = True
            st.rerun()

    if st.session_state.show_dialog:
        onboarding_dialog()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP HEADER
# ══════════════════════════════════════════════════════════════════════════════
gamma_used  = st.session_state.gamma_val
lambda_used = st.session_state.lambda_val
e_w = st.session_state.e_w
s_w = st.session_state.s_w
g_w = st.session_state.g_w

st.markdown("## GreenVest  |  ESG Portfolio Optimiser")
st.caption("Your investment profile is set. Configure your assets below — results update as you go.")

_, btn_col = st.columns([5,1])
with btn_col:
    if st.button("Update Profile"):
        st.session_state.is_update       = True
        st.session_state.onboarding_step = 1
        st.session_state.show_dialog     = True
        st.rerun()

if st.session_state.show_dialog and st.session_state.is_update:
    onboarding_dialog()

st.write("")
tab1, tab2, tab3 = st.tabs(["Inputs", "Results", "Charts"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INPUTS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Advanced parameter overrides ──────────────────────────────────────────
    with st.expander("Advanced Parameters  ·  Override profile-derived γ and λ", expanded=False):
        st.caption(
            "Your onboarding profile sets γ and λ automatically. "
            "Use the controls below to override them manually. "
            "Move the slider for a quick adjustment, or type an exact value in the field."
        )
        st.write("")

        # γ override
        st.markdown('<div class="override-label">Risk Aversion (γ)  ·  higher = more risk-averse</div>',
                    unsafe_allow_html=True)
        gc1, gc2 = st.columns([4,1])
        with gc1:
            st.slider(" ", 1.0, 10.0, st.session_state.gamma_val, 0.1,
                      key="_g_sl", on_change=_sync_gamma_slider,
                      label_visibility="collapsed",
                      help="γ > 1: risk-averse  |  γ = 1: risk-neutral  |  Negative γ is not used — it implies risk-seeking which is outside standard mean-variance theory.")
        with gc2:
            st.number_input("γ exact", 1.0, 10.0, st.session_state.gamma_val, 0.1,
                            key="_g_ni", on_change=_sync_gamma_input,
                            label_visibility="collapsed")
        gamma_used = st.session_state.gamma_val

        st.write("")

        # λ override
        st.markdown('<div class="override-label">ESG Preference (λ)  ·  higher = stronger sustainability bias</div>',
                    unsafe_allow_html=True)
        lc1, lc2 = st.columns([4,1])
        with lc1:
            st.slider(" ", 0.01, 0.20, st.session_state.lambda_val, 0.005,
                      key="_l_sl", on_change=_sync_lam_slider,
                      label_visibility="collapsed",
                      help="λ is the marginal utility of ESG score. Range 0.01–0.20 is economically calibrated so ESG adds meaningfully but does not dominate returns. E.g. λ=0.05, ESG=0.65 → adds 0.033 utility vs a return of ~0.08.")
        with lc2:
            st.number_input("λ exact", 0.01, 0.20, st.session_state.lambda_val, 0.005,
                            key="_l_ni", on_change=_sync_lam_input,
                            label_visibility="collapsed",
                            format="%.3f")
        lambda_used = st.session_state.lambda_val

        st.info(
            f"Active values:  **γ = {gamma_used}** (profile derived: {st.session_state.gamma})  "
            f"|  **λ = {lambda_used}** (profile derived: {st.session_state.lambda_esg})"
        )

    st.divider()

    # ── Market Parameters ──────────────────────────────────────────────────────
    st.subheader("Market Parameters")
    col1, col2 = st.columns(2)
    with col1:
        rf = st.number_input("Risk-free rate (%)",
                             min_value=0.0, max_value=20.0, value=4.5, step=0.1) / 100
    with col2:
        invest = st.number_input("Investment amount (GBP)",
                                 min_value=100.0, max_value=1_000_000.0,
                                 value=10_000.0, step=500.0)

    st.divider()

    # ── Excluded sectors (from onboarding) ────────────────────────────────────
    excluded_sectors = set()
    if st.session_state.excl_tobacco:  excluded_sectors.add("Tobacco")
    if st.session_state.excl_weapons:  excluded_sectors.add("Weapons / Defence")
    if st.session_state.excl_gambling: excluded_sectors.add("Gambling")
    if st.session_state.excl_fossil:   excluded_sectors.add("Fossil Fuels / Energy")
    if st.session_state.excl_alcohol:  excluded_sectors.add("Alcohol")

    # ── Asset inputs ───────────────────────────────────────────────────────────
    assets = {}
    for i in [1, 2]:
        st.subheader(f"Asset {i}")
        col_main, col_esg, col_hist = st.columns(3)

        with col_main:
            name   = st.text_input("Name", value=f"Asset {i}", key=f"name{i}")
            sector = st.selectbox(
                "Sector", SECTORS,
                index=0 if i==1 else 1, key=f"sector{i}",
                help="Select the sector this asset belongs to. "
                     "If it matches a sector you excluded, it will receive 0% allocation.",
            )
            mu    = st.number_input("Expected Return (%)", -100.0, 500.0,
                                    8.0 if i==1 else 5.0, 0.1, key=f"mu{i}") / 100
            sigma = st.number_input("Std Deviation (%)", 0.01, 500.0,
                                    15.0 if i==1 else 10.0, 0.1, key=f"sigma{i}") / 100

        with col_esg:
            st.caption("ESG Sub-scores (0–100)")
            e_score = st.slider("Environmental", 0.0, 100.0,
                                70.0 if i==1 else 50.0, key=f"e{i}")
            s_score = st.slider("Social",        0.0, 100.0,
                                65.0 if i==1 else 55.0, key=f"s{i}")
            g_score = st.slider("Governance",    0.0, 100.0,
                                60.0 if i==1 else 45.0, key=f"g{i}")

        with col_hist:
            esg_last = st.number_input("Last Year's ESG Score (0–100)",
                                       0.0, 100.0,
                                       65.0 if i==1 else 48.0,
                                       key=f"esg_last{i}")

        esg_c = composite_esg(e_score, s_score, g_score, e_w, s_w, g_w) / 100
        mom   = esg_momentum(
            composite_esg(e_score, s_score, g_score, e_w, s_w, g_w), esg_last)
        esg_a = min(esg_c + 0.05 * mom, 1.0)
        is_excluded = sector in excluded_sectors

        st.caption(
            f"Composite ESG: **{esg_c*100:.1f}** [{esg_rating(esg_c)}]"
            f"  ·  Momentum: {'+' if mom>=0 else ''}{mom*100:.1f}%"
            f"  ·  Adjusted: **{esg_a*100:.1f}**"
        )
        if esg_c >= 0.60 and g_score/100 < 0.35:
            st.warning(f"Greenwashing Alert: High overall ESG but low Governance on {name}.")
        if is_excluded:
            st.error(
                f"Sector excluded: {name} is classified as **{sector}**, "
                f"which you chose to exclude. This asset will receive **0% allocation**."
            )

        assets[i] = dict(
            name=name, sector=sector, mu=mu, sigma=sigma,
            e=e_score, s=s_score, g=g_score,
            esg_last=esg_last, esg_c=esg_c, mom=mom, esg_a=esg_a,
            is_excluded=is_excluded,
        )
        if i == 1:
            st.divider()

    st.divider()
    rho = st.slider("Correlation between assets",
                    -1.0, 1.0, 0.3, 0.01,
                    help="-1 = perfect inverse  |  0 = uncorrelated  |  1 = perfect positive")


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE  (sector exclusions as hard constraints)
# ══════════════════════════════════════════════════════════════════════════════
a1, a2 = assets[1], assets[2]

both_excluded = a1["is_excluded"] and a2["is_excluded"]
force_w1 = None
if a1["is_excluded"] and not a2["is_excluded"]:
    force_w1 = 0.0   # 100% asset 2
elif a2["is_excluded"] and not a1["is_excluded"]:
    force_w1 = 1.0   # 100% asset 1

if not both_excluded:
    w, mu_g, sg_g, eg_g, ut_g, ma_g, idx = optimise(
        a1["mu"], a2["mu"], a1["sigma"], a2["sigma"],
        a1["esg_a"], a2["esg_a"], rho, gamma_used, lambda_used,
        force_w1=force_w1,
    )
    w1 = w[idx]; w2 = 1 - w1
    mu_o = mu_g[idx]; sg_o = sg_g[idx]
    eg_o = eg_g[idx]; ut_o = ut_g[idx]
    sh_o   = (mu_o - rf) / sg_o if sg_o > 0 else float("nan")
    esg_sh = esg_sharpe(mu_o, rf, sg_o, lambda_used, eg_o)
    imp    = round(eg_o * 100, 1)
    td     = lambda_used * eg_o - gamma_used * sg_o

    # Comparison portfolios (only meaningful if both assets available)
    if force_w1 is None:
        _,mu_ms,sg_ms,eg_ms,_,_,idx_ms = optimise(
            a1["mu"],a2["mu"],a1["sigma"],a2["sigma"],
            a1["esg_a"],a2["esg_a"],rho,gamma_used,0.0)
        idx_mr = np.argmin(sg_g)
        idx_me = np.argmax(eg_g)
        mu50   = p_return(0.5,a1["mu"],a2["mu"])
        sg50   = p_std(0.5,a1["sigma"],a2["sigma"],rho)
        eg50   = p_esg(0.5,a1["esg_a"],a2["esg_a"])
        ret_cost = mu_ms[idx_ms]*100 - mu_o*100
    else:
        ret_cost = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    if both_excluded:
        st.error(
            "Both assets belong to excluded sectors. "
            "Please update your sector exclusions or asset selections in the Inputs tab."
        )
        st.stop()

    # ── Profile summary card ───────────────────────────────────────────────────
    excl_list = [s for s,f in [
        ("Tobacco",st.session_state.excl_tobacco),
        ("Weapons",st.session_state.excl_weapons),
        ("Gambling",st.session_state.excl_gambling),
        ("Fossil Fuels",st.session_state.excl_fossil),
        ("Alcohol",st.session_state.excl_alcohol),
    ] if f]
    excl_str = ", ".join(excl_list) if excl_list else "None"

    st.markdown(f"""
    <div class="summary-card">
      <div class="summary-title">Your Investment Profile</div>
      <table class="summary-table">
        <tr>
          <td class="summary-label">Risk Profile</td>
          <td class="summary-value">{st.session_state.profile}</td>
          <td class="summary-label">Risk Aversion (γ)</td>
          <td class="summary-value">{gamma_used}</td>
        </tr>
        <tr>
          <td class="summary-label">Investment Goal</td>
          <td class="summary-value">{st.session_state.goal_label}</td>
          <td class="summary-label">ESG Preference (λ)</td>
          <td class="summary-value">{lambda_used}</td>
        </tr>
        <tr>
          <td class="summary-label">ESG Priorities</td>
          <td class="summary-value">E: {e_w}/5 &nbsp; S: {s_w}/5 &nbsp; G: {g_w}/5</td>
          <td class="summary-label">Sector Exclusions</td>
          <td class="summary-value">{excl_str}</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

    if force_w1 is not None:
        excluded_name = a1["name"] if a1["is_excluded"] else a2["name"]
        held_name     = a2["name"] if a1["is_excluded"] else a1["name"]
        st.warning(
            f"Hard sector exclusion applied: **{excluded_name}** is in an excluded sector. "
            f"Portfolio is 100% **{held_name}**."
        )

    # ── Optimal Portfolio Summary ──────────────────────────────────────────────
    st.subheader("Optimal Portfolio Summary")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric(f"{a1['name']} weight", f"{w1*100:.2f}%")
    c2.metric(f"{a2['name']} weight", f"{w2*100:.2f}%")
    c3.metric("Expected Return",       f"{mu_o*100:.2f}%")
    c4.metric("Std Deviation",         f"{sg_o*100:.2f}%")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Sharpe Ratio",                   f"{sh_o:.4f}")
    c2.metric(f"ESG Score [{esg_rating(eg_o)}]",f"{eg_o*100:.1f} / 100")
    c3.metric("ESG-Adjusted Sharpe",            f"{esg_sh:.4f}")
    c4.metric("GreenVest Impact Score",         f"{imp:.1f}")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Trade-off Score", f"{td:.4f}")
    c2.metric("Utility Value",   f"{ut_o:.6f}")
    c3.metric("Risk Aversion γ", f"{gamma_used}")
    c4.metric("ESG Preference λ",f"{lambda_used}")

    # Why this portfolio is optimal
    high_esg_asset = a1["name"] if a1["esg_a"] > a2["esg_a"] else a2["name"]
    high_ret_asset = a1["name"] if a1["mu"] > a2["mu"] else a2["name"]
    gamma_desc = ("high risk aversion" if gamma_used >= 7
                  else "moderate risk aversion" if gamma_used >= 4
                  else "low risk aversion")
    lam_desc = ("strong ESG preference" if lambda_used >= 0.12
                else "moderate ESG preference" if lambda_used >= 0.06
                else "mild ESG preference")

    st.markdown(f"""
    <div class="insight-box">
      <strong>Why this is your optimal portfolio:</strong><br><br>
      With <strong>γ = {gamma_used}</strong> ({gamma_desc}), the optimiser penalises variance
      heavily — this explains the weighting toward lower-risk assets.
      With <strong>λ = {lambda_used}</strong> ({lam_desc}), ESG contributes
      meaningfully to utility but does not override financial considerations.
      The allocation of <strong>{w1*100:.1f}% in {a1["name"]}</strong> and
      <strong>{w2*100:.1f}% in {a2["name"]}</strong> maximises the utility function
      U = μ − (γ/2)σ² + λ·ESG across all possible two-asset combinations.
    </div>
    """, unsafe_allow_html=True)

    if force_w1 is None and ret_cost > 0.5:
        st.warning(
            f"ESG cost: your sustainability preference costs {ret_cost:.2f}% in expected return "
            f"vs the financial-only optimal. This is the price of internalising ESG externalities."
        )
    elif force_w1 is None and ret_cost <= 0:
        st.success(
            f"ESG pays off here: your ESG-optimal portfolio outperforms the "
            f"financial-only benchmark by {abs(ret_cost):.2f}%."
        )

    st.divider()

    # ── Financial Interpretation ───────────────────────────────────────────────
    st.subheader("What This Means  ·  ESG and Financial Performance")

    if force_w1 is None:
        tangency_return = mu_ms[idx_ms]*100
        tangency_sharpe = ((mu_ms[idx_ms] - rf) / sg_ms[idx_ms]
                           if sg_ms[idx_ms] > 0 else float("nan"))
        esg_opt_sharpe  = sh_o
        sharpe_diff     = esg_opt_sharpe - tangency_sharpe

        st.markdown(f"""
        <div class="insight-box">
          <strong>Tangency portfolio vs your ESG-optimal portfolio:</strong><br><br>
          The <strong>tangency portfolio</strong> (λ=0, purely financial) has an expected
          return of <strong>{tangency_return:.2f}%</strong> and a Sharpe ratio of
          <strong>{tangency_sharpe:.4f}</strong>.<br>
          Your <strong>ESG-optimal portfolio</strong> has a Sharpe ratio of
          <strong>{esg_opt_sharpe:.4f}</strong> — a difference of
          <strong>{sharpe_diff:+.4f}</strong>.<br><br>
          This trade-off reflects a core principle of sustainable finance:
          investors who incorporate ESG accept a modest reduction in risk-adjusted returns
          in exchange for portfolios that internalise environmental and social externalities.
          This is not irrational — it reflects genuine preferences and long-term risk management.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
      <strong>Why investors choose ESG portfolios:</strong><br><br>
      <strong>1. Internalisation of externalities</strong> — ESG investors account for
      environmental and social costs that traditional finance ignores. By penalising
      low-ESG assets, they reflect the true long-run cost of carbon emissions,
      poor governance, or labour violations.<br><br>
      <strong>2. Long-term risk management</strong> — Companies with poor ESG profiles
      carry higher regulatory, reputational, and operational risk. ESG integration
      is partly a forward-looking risk screen, not just an ethical stance.<br><br>
      <strong>3. Preference alignment</strong> — The parameter λ in the utility function
      directly captures an investor's willingness to sacrifice return for sustainability.
      Your λ = {lam} means you are willing to give up approximately {sacrifice:.2f}%
      in expected return for a one-point improvement in ESG score (on a 0–1 scale).
    </div>
    """.format(lam=lambda_used, sacrifice=lambda_used*100), unsafe_allow_html=True)

    st.divider()

    # ── ESG Breakdown ──────────────────────────────────────────────────────────
    st.subheader("ESG Breakdown")
    esg_df = pd.DataFrame({
        "Pillar":      ["Environmental","Social","Governance","Composite","Momentum-adj"],
        a1["name"]:    [a1["e"],a1["s"],a1["g"],
                        f"{a1['esg_c']*100:.1f}",f"{a1['esg_a']*100:.1f}"],
        a2["name"]:    [a2["e"],a2["s"],a2["g"],
                        f"{a2['esg_c']*100:.1f}",f"{a2['esg_a']*100:.1f}"],
        "Your Weight": [f"{e_w}/5",f"{s_w}/5",f"{g_w}/5","—","—"],
    })
    st.dataframe(esg_df, use_container_width=True, hide_index=True)

    if force_w1 is None:
        st.divider()
        st.subheader("Portfolio Comparison")
        comp_df = pd.DataFrame({
            "Portfolio": ["GreenVest Optimal","Tangency (λ=0)","Min Risk","Max ESG","50 / 50"],
            f"W({a1['name'][:10]})": [
                f"{w1*100:.1f}%",         f"{w[idx_ms]*100:.1f}%",
                f"{w[idx_mr]*100:.1f}%",  f"{w[idx_me]*100:.1f}%",  "50.0%",
            ],
            "Return": [
                f"{mu_o*100:.2f}%",         f"{mu_ms[idx_ms]*100:.2f}%",
                f"{mu_g[idx_mr]*100:.2f}%", f"{mu_g[idx_me]*100:.2f}%",
                f"{mu50*100:.2f}%",
            ],
            "Std Dev": [
                f"{sg_o*100:.2f}%",         f"{sg_ms[idx_ms]*100:.2f}%",
                f"{sg_g[idx_mr]*100:.2f}%", f"{sg_g[idx_me]*100:.2f}%",
                f"{sg50*100:.2f}%",
            ],
            "Sharpe": [
                f"{sh_o:.4f}",
                f"{((mu_ms[idx_ms]-rf)/sg_ms[idx_ms] if sg_ms[idx_ms]>0 else float('nan')):.4f}",
                f"{((mu_g[idx_mr]-rf)/sg_g[idx_mr] if sg_g[idx_mr]>0 else float('nan')):.4f}",
                f"{((mu_g[idx_me]-rf)/sg_g[idx_me] if sg_g[idx_me]>0 else float('nan')):.4f}",
                f"{((mu50-rf)/sg50 if sg50>0 else float('nan')):.4f}",
            ],
            "ESG Score": [
                f"{eg_o*100:.1f}",        f"{eg_ms[idx_ms]*100:.1f}",
                f"{eg_g[idx_mr]*100:.1f}",f"{eg_g[idx_me]*100:.1f}",
                f"{eg50*100:.1f}",
            ],
            "Rating": [
                esg_rating(eg_o),         esg_rating(eg_ms[idx_ms]),
                esg_rating(eg_g[idx_mr]), esg_rating(eg_g[idx_me]),
                esg_rating(eg50),
            ],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader(f"Future Value  ·  GBP {invest:,.0f} invested")
        fv_df = pd.DataFrame({
            "Horizon":   ["5 years","10 years","20 years","30 years"],
            "GreenVest": [f"GBP {future_value(invest,mu_o,y):,.0f}"   for y in [5,10,20,30]],
            "50 / 50":   [f"GBP {future_value(invest,mu50,y):,.0f}"   for y in [5,10,20,30]],
            "Difference":[f"GBP {future_value(invest,mu_o,y)-future_value(invest,mu50,y):+,.0f}"
                          for y in [5,10,20,30]],
        })
        st.dataframe(fv_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Sensitivity  ·  Optimal allocation as λ changes")
        sens_rows = []
        for lam_i in np.linspace(0.01, 0.20, 11):
            _,mi,si,ei,_,_,xi = optimise(
                a1["mu"],a2["mu"],a1["sigma"],a2["sigma"],
                a1["esg_a"],a2["esg_a"],rho,gamma_used,lam_i)
            sens_rows.append({
                "λ":                     f"{lam_i:.3f}",
                f"W({a1['name'][:10]})": f"{w[xi]*100:.1f}%",
                "Return":                f"{mi[xi]*100:.2f}%",
                "ESG Score":             f"{ei[xi]*100:.1f}",
                "Rating":                esg_rating(ei[xi]),
            })
        st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:

    if both_excluded:
        st.error("Both assets excluded — no portfolio to chart.")
        st.stop()

    BG   = "#f4faf5"
    GRID = "#c3dfc9"
    GRN  = "#1e5239"
    GRN2 = "#2d6a4f"
    AMB  = "#b87333"
    BLU  = "#3a5fb8"

    def style(ax):
        ax.set_facecolor(BG)
        ax.grid(True, alpha=0.2, color=GRID)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.tick_params(colors="#3d5e47", labelsize=8)
        ax.xaxis.label.set_color("#3d5e47")
        ax.yaxis.label.set_color("#3d5e47")

    # ── Row 1: Portfolio Construction ─────────────────────────────────────────
    st.markdown('<div class="chart-section">Portfolio Construction</div>',
                unsafe_allow_html=True)

    if force_w1 is None:
        fig1, (ax1a, ax1b, ax1c) = plt.subplots(1, 3, figsize=(15, 4.5))
        fig1.patch.set_facecolor(BG)

        # Chart 1 — ESG-coloured frontier
        style(ax1a)
        pts  = np.array([sg_g, ma_g]).T.reshape(-1,1,2)
        segs = np.concatenate([pts[:-1],pts[1:]],axis=1)
        lc   = LineCollection(segs, cmap="YlGn", linewidth=2.5)
        lc.set_array(eg_g); ax1a.add_collection(lc); ax1a.autoscale()
        ax1a.plot(sg_g, mu_g, color=GRID, linewidth=1.2, linestyle="--",
                  label="Traditional frontier")
        fig1.colorbar(lc, ax=ax1a, shrink=0.85, label="ESG Score")
        ax1a.scatter(sg_o,          ma_g[idx],    color=GRN, s=110,
                     marker="D", label="GreenVest Opt", zorder=5, edgecolors="#7ec8a0")
        ax1a.scatter(sg_ms[idx_ms], ma_g[idx_ms], color=AMB, s=70,
                     marker="^", label="Tangency (λ=0)", zorder=4)
        ax1a.scatter(sg_g[idx_mr],  ma_g[idx_mr], color=BLU, s=70,
                     marker="s", label="Min Risk", zorder=4)
        ax1a.set_xlabel("Std Dev"); ax1a.set_ylabel("ESG-Adjusted Return")
        ax1a.set_title("ESG-Efficient Frontier", color=GRN, fontweight="bold", pad=10)
        ax1a.legend(fontsize=7)

        # Chart 2 — Utility curve
        style(ax1b)
        ax1b.plot(w*100, ut_g, color=GRN2, linewidth=2)
        ax1b.axvline(w1*100, color=GRN, linestyle="--", alpha=0.6,
                     label=f"Optimal: {w1*100:.1f}%")
        ax1b.scatter(w1*100, ut_o, color=GRN, s=90, marker="D", zorder=5)
        ax1b.set_xlabel(f"Weight in {a1['name']} (%)")
        ax1b.set_ylabel("Utility  U = μ − (γ/2)σ² + λ·ESG")
        ax1b.set_title("Utility vs Portfolio Weight", color=GRN, fontweight="bold", pad=10)
        ax1b.legend(fontsize=7)

        # Chart 3 — Future value
        style(ax1c)
        yr = np.arange(0,31)
        ax1c.plot(yr,[future_value(invest,mu_o, y) for y in yr],
                  color=GRN, linewidth=2.5, label=f"GreenVest ({mu_o*100:.1f}%)")
        ax1c.plot(yr,[future_value(invest,mu50,  y) for y in yr],
                  color=AMB, linewidth=1.8, linestyle="--", label=f"50/50 ({mu50*100:.1f}%)")
        ax1c.plot(yr,[future_value(invest,rf,    y) for y in yr],
                  color="#999", linewidth=1.2, linestyle=":", label=f"Risk-free ({rf*100:.1f}%)")
        ax1c.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"GBP{x/1000:.0f}k"))
        ax1c.set_xlabel("Years"); ax1c.set_title("Future Value", color=GRN,
                                                  fontweight="bold", pad=10)
        ax1c.legend(fontsize=7)

        plt.tight_layout(pad=2.0)
        st.pyplot(fig1); plt.close(fig1)

        # ── Row 2: Sensitivity Analysis ────────────────────────────────────────
        st.write("")
        st.markdown('<div class="chart-section">Sensitivity Analysis  ·  How λ Shapes the Portfolio</div>',
                    unsafe_allow_html=True)

        lam_r = np.linspace(0.01, 0.20, 60)
        w1_s, mu_s, esg_s = [], [], []
        for lam_i in lam_r:
            _,mi,si,ei,_,_,xi = optimise(
                a1["mu"],a2["mu"],a1["sigma"],a2["sigma"],
                a1["esg_a"],a2["esg_a"],rho,gamma_used,lam_i)
            w1_s.append(w[xi]*100); mu_s.append(mi[xi]*100); esg_s.append(ei[xi]*100)

        fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(15, 4.5))
        fig2.patch.set_facecolor(BG)

        for ax, data, ylabel, color, title in [
            (ax2a, w1_s,  f"W({a1['name'][:10]}) %", GRN2, "Sensitivity: Allocation"),
            (ax2b, mu_s,  "Return (%)",               AMB,  "Sensitivity: Return"),
            (ax2c, esg_s, "ESG Score",                GRN,  "Sensitivity: ESG"),
        ]:
            style(ax)
            ax.plot(lam_r, data, color=color, linewidth=2)
            ax.axvline(lambda_used, color="#888", linestyle="--", alpha=0.7,
                       label=f"Your λ = {lambda_used}")
            ax.set_xlabel("ESG Preference (λ)")
            ax.set_ylabel(ylabel)
            ax.set_title(title, color=GRN, fontweight="bold", pad=10)
            ax.legend(fontsize=7)

        plt.tight_layout(pad=2.0)
        st.pyplot(fig2); plt.close(fig2)

    else:
        # Single-asset case — just show a simple summary chart
        st.info("One asset is excluded. Showing single-asset projection only.")
        fig_s, ax_s = plt.subplots(figsize=(7, 4))
        fig_s.patch.set_facecolor(BG); style(ax_s)
        yr = np.arange(0,31)
        ax_s.plot(yr,[future_value(invest,mu_o,y) for y in yr],
                  color=GRN,linewidth=2.5,label=f"{a1['name'] if w1==1 else a2['name']}")
        ax_s.plot(yr,[future_value(invest,rf,y) for y in yr],
                  color="#999",linewidth=1.2,linestyle=":",label=f"Risk-free ({rf*100:.1f}%)")
        ax_s.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"GBP{x/1000:.0f}k"))
        ax_s.set_xlabel("Years"); ax_s.set_title("Future Value", color=GRN, fontweight="bold")
        ax_s.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig_s); plt.close(fig_s)