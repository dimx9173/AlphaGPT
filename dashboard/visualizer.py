import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_pnl_distribution(portfolio_df):
    if portfolio_df.empty:
        return go.Figure()
    
    colors = ['#00FF00' if x > 0 else '#FF0000' for x in portfolio_df['pnl_pct']]
    
    fig = go.Figure(data=[go.Bar(
        x=portfolio_df['symbol'],
        y=portfolio_df['pnl_pct'],
        marker_color=colors
    )])
    
    fig.update_layout(
        title="Current Positions PnL %",
        yaxis_tickformat='.2%',
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def plot_market_scatter(market_df):
    if market_df.empty: return go.Figure()
    
    fig = px.scatter(
        market_df,
        x="liquidity",
        y="volume",
        size="fdv",
        color="symbol",
        hover_name="symbol",
        log_x=True,
        log_y=True,
        title="Market Liquidity vs Volume (Bubble Size = FDV)",
        template="plotly_dark"
    )
    return fig

def plot_training_curve(history):
    if history is None:
        return go.Figure()
    try:
        if isinstance(history, dict) and "step" in history:
            x = history.get("step") or []
            y_reward = history.get("avg_reward") or []
            y_best = history.get("best_score") or []
            y_sharpe = history.get("sharpe") or []
            fig = go.Figure()
            if x and y_reward:
                fig.add_trace(go.Scatter(x=x, y=y_reward, mode="lines", name="avg_reward"))
            if x and y_best:
                fig.add_trace(go.Scatter(x=x, y=y_best, mode="lines", name="best_score"))
            if x and y_sharpe:
                fig.add_trace(go.Scatter(x=x, y=y_sharpe, mode="lines", name="sharpe", yaxis="y2"))
            fig.update_layout(title="Training Curve (reward / best_score / sharpe)", template="plotly_dark", margin=dict(l=20,r=20,t=40,b=20), yaxis2=dict(overlaying="y", side="right", title="sharpe"))
            return fig
        if isinstance(history, list):
            steps = [r.get("step") for r in history if isinstance(r, dict)]
            rewards = [r.get("avg_reward") for r in history if isinstance(r, dict)]
            sharpes = [r.get("sharpe") for r in history if isinstance(r, dict)]
            fig = go.Figure()
            if steps and rewards:
                fig.add_trace(go.Scatter(x=steps, y=rewards, mode="lines", name="avg_reward"))
            if steps and sharpes and any(v is not None for v in sharpes):
                fig.add_trace(go.Scatter(x=steps, y=[v if v is not None else 0 for v in sharpes], mode="lines", name="sharpe", yaxis="y2"))
            fig.update_layout(title="Training Curve (from metrics.jsonl)", template="plotly_dark", margin=dict(l=20,r=20,t=40,b=20), yaxis2=dict(overlaying="y", side="right", title="sharpe"))
            return fig
    except Exception:
        return go.Figure()
    return go.Figure()

def plot_walk_forward(history):
    if not isinstance(history, dict):
        return go.Figure()
    try:
        steps = history.get("step") or []
        sharpe = history.get("sharpe") or []
        max_dd = history.get("max_dd") or []
        if not steps or not sharpe:
            return go.Figure()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=sharpe, mode="lines+markers", name="Sharpe"))
        if max_dd and len(max_dd)==len(steps):
            fig.add_trace(go.Scatter(x=steps, y=max_dd, mode="lines", name="MaxDD"))
        fig.update_layout(title="Walk-forward Sharpe / MaxDD", template="plotly_dark", margin=dict(l=20,r=20,t=40,b=20))
        return fig
    except Exception:
        return go.Figure()