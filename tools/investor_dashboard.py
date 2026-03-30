"""
Investor Dashboard — Interactive Streamlit Application.

I provide a professional, interactive dashboard for presenting
EntsoDRL trading performance to investors and stakeholders.

Features:
- Real-time charts with zoom/pan
- Per-market revenue breakdown
- Daily P&L analysis with win rate
- SoC management visualization
- Trade-by-trade analysis
- Export to CSV

Usage:
    streamlit run tools/investor_dashboard.py -- --data evaluation_results/eval_data.csv

Requirements:
    pip install streamlit plotly
"""

import sys
sys.path.insert(0, '.')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
from pathlib import Path


# ─── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="EntsoDRL Trading Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """I load and cache the evaluation data."""
    return pd.read_csv(path)


def compute_metrics(df: pd.DataFrame, time_step: float) -> dict:
    """I compute key performance metrics."""
    total_days = len(df) * time_step / 24
    total_profit = df['total_profit'].iloc[-1]
    total_cycles = df['total_cycles'].iloc[-1]

    steps_per_day = int(24 / time_step)
    daily_pnl = []
    for d in range(int(total_days)):
        day_slice = df.iloc[d * steps_per_day:(d + 1) * steps_per_day]
        if len(day_slice) > 0:
            daily_pnl.append(day_slice['net_profit_step'].sum())

    daily_pnl = np.array(daily_pnl) if daily_pnl else np.array([0])
    win_days = (daily_pnl > 0).sum()

    sharpe = 0
    if len(daily_pnl) > 1 and np.std(daily_pnl) > 0:
        sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(365)

    return {
        'total_profit': total_profit,
        'daily_avg': total_profit / max(total_days, 1),
        'annualized': total_profit / max(total_days, 1) * 365,
        'total_days': total_days,
        'total_cycles': total_cycles,
        'daily_cycles': total_cycles / max(total_days, 1),
        'mean_soc': df['soc'].mean(),
        'min_soc': df['soc'].min(),
        'max_soc': df['soc'].max(),
        'win_rate': win_days / max(len(daily_pnl), 1),
        'win_days': win_days,
        'total_eval_days': len(daily_pnl),
        'sharpe': sharpe,
        'daily_pnl': daily_pnl,
        'dam_profit': df['dam_profit'].iloc[-1],
        'intraday_profit': df['intraday_profit'].iloc[-1],
        'afrr_cap_profit': df['afrr_cap_profit'].iloc[-1],
        'afrr_energy_profit': df['afrr_energy_profit'].iloc[-1],
        'mfrr_profit': df['mfrr_profit'].iloc[-1],
    }


def main():
    st.title("⚡ EntsoDRL Battery Trading System")
    st.markdown("### AI-Powered Energy Trading Performance Dashboard")

    # ─── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.header("Configuration")
        data_path = st.text_input("Data Path", "evaluation_results/eval_data.csv")
        time_step = st.selectbox("Time Resolution", [0.25, 1.0], index=0,
                                format_func=lambda x: f"{int(x*60)} minutes")

        if not Path(data_path).exists():
            st.error(f"File not found: {data_path}")
            st.info("Run evaluation first:\n```\npython tools/evaluation_visualizer.py --model <path>\n```")
            return

    # I load data
    df = load_data(data_path)
    metrics = compute_metrics(df, time_step)
    hours = df['step'] * time_step

    # ─── KPI Cards ───────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Profit", f"{metrics['total_profit']:,.0f} EUR")
    with col2:
        st.metric("Daily Average", f"{metrics['daily_avg']:,.0f} EUR")
    with col3:
        st.metric("Annualized", f"{metrics['annualized']:,.0f} EUR")
    with col4:
        st.metric("Win Rate", f"{metrics['win_rate']:.0%}")
    with col5:
        st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")

    st.markdown("---")

    # ─── Tab Layout ──────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", "🔋 Battery", "💰 Revenue", "📈 Trades", "📋 Raw Data"
    ])

    with tab1:
        # I create overview with SoC + Prices
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=('Battery State of Charge', 'DAM Price + Trading Actions'),
                           vertical_spacing=0.08, row_heights=[0.4, 0.6])

        # SoC
        fig.add_trace(go.Scatter(x=hours, y=df['soc'] * 100, mode='lines',
                                fill='tozeroy', fillcolor='rgba(70,130,180,0.2)',
                                line=dict(color='steelblue', width=1.5),
                                name='SoC %'), row=1, col=1)
        fig.add_hline(y=5, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
        fig.add_hline(y=95, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)

        # Prices
        fig.add_trace(go.Scatter(x=hours, y=df['dam_price'], mode='lines',
                                line=dict(color='navy', width=1), name='DAM Price'), row=2, col=1)

        # Buy/Sell markers
        sells = df[df['xbid_mw'] > 1.0]
        buys = df[df['xbid_mw'] < -1.0]
        if len(sells) > 0:
            fig.add_trace(go.Scatter(x=sells['step'] * time_step, y=sells['dam_price'],
                                    mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'),
                                    name=f'Sell ({len(sells)}x)'), row=2, col=1)
        if len(buys) > 0:
            fig.add_trace(go.Scatter(x=buys['step'] * time_step, y=buys['dam_price'],
                                    mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'),
                                    name=f'Buy ({len(buys)}x)'), row=2, col=1)

        fig.update_layout(height=600, showlegend=True)
        fig.update_yaxes(title_text="SoC (%)", range=[0, 100], row=1, col=1)
        fig.update_yaxes(title_text="EUR/MWh", row=2, col=1)
        fig.update_xaxes(title_text="Hours", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("SoC Statistics")
            st.metric("Mean SoC", f"{metrics['mean_soc']:.0%}")
            st.metric("Min SoC", f"{metrics['min_soc']:.0%}")
            st.metric("Max SoC", f"{metrics['max_soc']:.0%}")
            st.metric("Total Cycles", f"{metrics['total_cycles']:.1f}")
            st.metric("Cycles/Day", f"{metrics['daily_cycles']:.2f}")

        with col2:
            # I create SoC histogram
            fig_hist = go.Figure(data=[go.Histogram(x=df['soc'] * 100, nbinsx=50,
                                                     marker_color='steelblue')])
            fig_hist.update_layout(title="SoC Distribution", xaxis_title="SoC (%)",
                                   yaxis_title="Frequency", height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)

        with col1:
            # I create revenue pie chart
            market_data = {
                'DAM': metrics['dam_profit'],
                'IntraDay': metrics['intraday_profit'],
                'aFRR Cap': metrics['afrr_cap_profit'],
                'aFRR Energy': metrics['afrr_energy_profit'],
                'mFRR': metrics['mfrr_profit'],
            }
            positive = {k: v for k, v in market_data.items() if v > 0}
            negative = {k: v for k, v in market_data.items() if v <= 0}

            if positive:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(positive.keys()),
                    values=list(positive.values()),
                    hole=0.4,
                    textinfo='label+value',
                    texttemplate='%{label}<br>%{value:,.0f} EUR'
                )])
                fig_pie.update_layout(title="Revenue by Market (Positive)", height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

            if negative:
                st.warning("Markets with negative P&L:")
                for k, v in negative.items():
                    st.write(f"  {k}: {v:,.0f} EUR")

        with col2:
            # I create daily P&L bar chart
            daily = metrics['daily_pnl']
            colors = ['#2ecc71' if p > 0 else '#e74c3c' for p in daily]
            fig_daily = go.Figure(data=[go.Bar(x=list(range(len(daily))),
                                               y=daily, marker_color=colors)])
            fig_daily.add_hline(y=0, line_color="black")
            fig_daily.add_hline(y=np.mean(daily), line_dash="dash", line_color="navy",
                               annotation_text=f"Avg: {np.mean(daily):,.0f} EUR")
            fig_daily.update_layout(title="Daily P&L", xaxis_title="Day",
                                    yaxis_title="EUR", height=400)
            st.plotly_chart(fig_daily, use_container_width=True)

    with tab4:
        st.subheader("Hourly Trading Pattern")

        # I create hourly heatmap
        steps_per_hour = int(1 / time_step)
        df['hour_of_day'] = ((df['step'] * time_step) % 24).astype(int)
        hourly_xbid = df.groupby('hour_of_day')['xbid_mw'].mean()

        fig_hourly = go.Figure(data=[go.Bar(
            x=hourly_xbid.index,
            y=hourly_xbid.values,
            marker_color=['#2ecc71' if v < 0 else '#e74c3c' if v > 0 else '#bdc3c7'
                          for v in hourly_xbid.values]
        )])
        fig_hourly.update_layout(
            title="Average XBID MW by Hour (negative=buy, positive=sell)",
            xaxis_title="Hour of Day", yaxis_title="MW",
            height=400
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

        # I show action distribution
        col1, col2 = st.columns(2)
        with col1:
            xbid_sell = (df['xbid_mw'] > 0.5).sum()
            xbid_buy = (df['xbid_mw'] < -0.5).sum()
            xbid_idle = len(df) - xbid_sell - xbid_buy
            fig_dist = go.Figure(data=[go.Pie(
                labels=['Sell', 'Buy', 'Idle'],
                values=[xbid_sell, xbid_buy, xbid_idle],
                marker_colors=['#e74c3c', '#2ecc71', '#bdc3c7']
            )])
            fig_dist.update_layout(title="XBID Action Distribution", height=350)
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            sells_df = df[df['xbid_mw'] > 0.5]
            buys_df = df[df['xbid_mw'] < -0.5]
            st.markdown("**Trading Statistics**")
            if len(sells_df) > 0:
                st.write(f"Sell: {len(sells_df)} trades, avg {sells_df['xbid_mw'].mean():.1f} MW "
                        f"at avg price {sells_df['dam_price'].mean():.1f} EUR")
            if len(buys_df) > 0:
                st.write(f"Buy: {len(buys_df)} trades, avg {buys_df['xbid_mw'].abs().mean():.1f} MW "
                        f"at avg price {buys_df['dam_price'].mean():.1f} EUR")
            overall_mean = df['dam_price'].mean()
            st.write(f"Overall mean price: {overall_mean:.1f} EUR/MWh")

    with tab5:
        st.subheader("Raw Evaluation Data")
        st.dataframe(df, use_container_width=True, height=500)
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "eval_data.csv", "text/csv")

    # ─── Footer ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        f"*EntsoDRL v6 | Battery: 146 MWh / 30 MW | "
        f"Evaluation: {metrics['total_days']:.1f} days | "
        f"Generated by AI Trading System*"
    )


if __name__ == '__main__':
    main()
