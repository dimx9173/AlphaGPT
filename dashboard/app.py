# Secure run: streamlit run dashboard/app.py --server.address 127.0.0.1 --server.port 8501
# For production, bind to 127.0.0.1 and expose via reverse proxy (nginx/caddy) with TLS and auth; never expose directly to public internet.
import os
import streamlit as st
import pandas as pd
import time
from data_service import DashboardService
from visualizer import plot_pnl_distribution, plot_market_scatter, plot_training_curve, plot_walk_forward

st.set_page_config(
    page_title="MemeAlpha Commander",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .stDataFrame { border: none; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_service():
    return DashboardService()

svc = get_service()

st.sidebar.title("MemeAlpha Bot")
st.sidebar.markdown("---")

with st.sidebar:
    st.subheader("Wallet Status")
    bal = svc.get_wallet_balance()
    st.metric("SOL Balance", f"{bal:.4f} SOL")
    
    st.markdown("---")
    st.subheader("Control Panel")
    if st.button("Refresh Data"):
        st.rerun()
        
    dashboard_token = os.getenv("DASHBOARD_TOKEN")
    if not dashboard_token:
        st.warning("DASHBOARD_TOKEN not set — dashboard should be bound to 127.0.0.1. Set DASHBOARD_TOKEN to protect EMERGENCY STOP.")
        confirm_stop = st.checkbox("I confirm EMERGENCY STOP")
        token_input = None
    else:
        token_input = st.text_input("DASHBOARD_TOKEN", type="password")
        confirm_stop = False

    if st.button("EMERGENCY STOP", type="primary"):
        if dashboard_token:
            if token_input != dashboard_token:
                st.error("Invalid token — EMERGENCY STOP not sent.")
            else:
                import os as _os
                with open(_os.getenv("STOP_SIGNAL_PATH", "STOP_SIGNAL"), "w") as f:
                    f.write("STOP")
                st.error("STOP SIGNAL SENT, Process will terminate on next cycle.")
        else:
            if not confirm_stop:
                st.error("Please check 'I confirm EMERGENCY STOP' to proceed.")
            else:
                import os as _os
                with open(_os.getenv("STOP_SIGNAL_PATH", "STOP_SIGNAL"), "w") as f:
                    f.write("STOP")
                st.error("STOP SIGNAL SENT, Process will terminate on next cycle.")

col1, col2, col3, col4 = st.columns(4)
portfolio_df = svc.load_portfolio()
market_df = svc.get_market_overview()
strategy_data = svc.load_strategy_info()

open_positions = len(portfolio_df)
total_invested = portfolio_df['initial_cost_sol'].sum() if not portfolio_df.empty else 0.0

with col1:
    st.metric("Open Positions", f"{open_positions} / 5")
with col2:
    st.metric("Total Invested", f"{total_invested:.2f} SOL")
with col3:
    if not portfolio_df.empty:
        current_val = (portfolio_df['amount_held'] * portfolio_df['highest_price']).sum()
        pnl_sol = current_val - total_invested
        st.metric("Unrealized PnL (Est)", f"{pnl_sol:+.3f} SOL", delta_color="normal")
    else:
        st.metric("Unrealized PnL", "0.00 SOL")
with col4:
    st.metric("Active Strategy", "AlphaGPT-v1", help=str(strategy_data))

try:
    tm = svc.get_training_metrics()
    ps = svc.get_pipeline_status()
except Exception:
    tm = {"latest": None, "history": None, "source": None}
    ps = {}
latest = (tm or {}).get("latest") or {}
hist = (tm or {}).get("history")
def _fmt(v, fmt="{:.3f}"):
    try:
        if v is None: return "—"
        return fmt.format(float(v))
    except Exception:
        return str(v)
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Sharpe (last)", _fmt(latest.get("sharpe"), "{:.2f}"), help=f"source: {(tm or {}).get('source')}")
with m2:
    st.metric("MaxDD (last)", _fmt(latest.get("max_dd"), "{:.4f}"))
with m3:
    st.metric("Turnover (last)", _fmt(latest.get("turnover"), "{:.4f}"))
with m4:
    lu = ps.get("last_updated") if isinstance(ps, dict) else None
    st.metric("Pipeline", f"{ps.get('token_count', '—')} tokens", delta=str(lu)[:19] if lu else None, help=str({k: ps.get(k) for k in ("ohlcv_count","checkpoint_exists","metrics.jsonl","training_history.json") if isinstance(ps, dict)}))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Portfolio", "Market Scanner", "Training Curve", "Pipeline Status", "Logs"])

with tab1:
    st.subheader("Active Holdings")
    if not portfolio_df.empty:
        # P4: ensure venue columns exist for old state files
        for _c, _d in (("venue", "solana"), ("side", "LONG"), ("leverage", 1.0)):
            if _c not in portfolio_df.columns:
                portfolio_df[_c] = _d
        venues = sorted(portfolio_df["venue"].fillna("solana").unique().tolist())
        venue_sel = st.multiselect("Venue filter", venues, default=venues)
        filt_df = portfolio_df[portfolio_df["venue"].isin(venue_sel)] if venue_sel else portfolio_df
        # Display Table
        display_cols = [c for c in ['venue', 'side', 'leverage', 'symbol', 'entry_price',
                                    'highest_price', 'amount_held', 'pnl_pct', 'is_moonbag']
                        if c in filt_df.columns]

        # Format for display
        show_df = filt_df[display_cols].copy()
        if 'pnl_pct' in show_df.columns:
            show_df['pnl_pct'] = show_df['pnl_pct'].apply(lambda x: f"{x:.2%}")
        if 'entry_price' in show_df.columns:
            show_df['entry_price'] = show_df['entry_price'].apply(lambda x: f"{x:.6f}")

        st.dataframe(show_df, use_container_width=True, hide_index=True)
        st.caption(f"By venue: {filt_df.groupby('venue').size().to_dict()}" if not filt_df.empty else "")

        # Display Chart
        st.plotly_chart(plot_pnl_distribution(filt_df), use_container_width=True)
    else:
        st.info("No active positions. The bot is scanning...")

with tab2:
    st.subheader("Top Opportunities (DB Snapshot)")
    if not market_df.empty:
        st.plotly_chart(plot_market_scatter(market_df), use_container_width=True)
        st.dataframe(market_df, use_container_width=True)
    else:
        st.warning("No market data found in DB. Is the Data Pipeline running?")

with tab3:
    st.subheader("Training Curve (Sharpe / Reward)")
    if hist is not None:
        st.plotly_chart(plot_training_curve(hist), use_container_width=True)
        if isinstance(hist, dict) and hist.get("step"):
            st.plotly_chart(plot_walk_forward(hist), use_container_width=True)
        elif isinstance(hist, list):
            st.caption(f"History from {(tm or {}).get('source')} — {len(hist)} points")
            st.plotly_chart(plot_training_curve(hist), use_container_width=True)
    else:
        st.info("No training metrics yet. Run training to generate training_history.json / metrics.jsonl (Sharpe/MaxDD/Turnover).")
        st.caption("Tip: cat training_history.json or cat metrics.jsonl to verify export.")

with tab4:
    st.subheader("Pipeline Status")
    if isinstance(ps, dict):
        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Tokens", str(ps.get("token_count", "—")))
            st.metric("OHLCV candles", str(ps.get("ohlcv_count", "—")))
        with cB:
            st.metric("Last Updated (DB)", str(ps.get("last_updated", "—"))[:19] if ps.get("last_updated") else "—")
            st.metric("Checkpoint", "exists" if ps.get("checkpoint_exists") else "missing")
        with cC:
            st.metric("metrics.jsonl", "exists" if ps.get("metrics.jsonl") else "missing")
            st.metric("training_history.json", "exists" if ps.get("training_history.json") else "missing")
        st.json(ps)
    else:
        st.warning("Pipeline status unavailable (DB not reachable). File-based metrics still shown above.")

with tab5:
    st.subheader("System Logs (Tail 20)")
    logs = svc.get_recent_logs(20)
    if logs:
        st.code("".join(logs), language="text")
    else:
        st.caption("No logs found or log file path incorrect.")

time.sleep(1) 
if st.checkbox("Auto-Refresh (30s)", value=True):
    time.sleep(30)
    st.rerun()