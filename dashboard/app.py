"""Electricity demand monitoring and model evaluation."""

from datetime import date
from pathlib import Path
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.client import ForecastClient
from src.metrics import evaluate_records

load_dotenv()
st.set_page_config(page_title="Türkiye Electricity Forecast", page_icon="⚡", layout="wide")
st.markdown(
    """
<style>
.block-container { max-width: 1400px; padding-top: 3.5rem; }
h1, h2, h3 { color: #172c46; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #e3e8ef; border-radius: 10px; padding: 16px; }
[data-testid="stMetricLabel"] { color: #54657b; }
[data-testid="stMetricValue"] { color: #172c46; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def fetch(route, **params):
    def setting(name):
        if os.getenv(name):
            return os.getenv(name)
        try:
            return st.secrets.get(name)
        except FileNotFoundError:
            return None

    client = ForecastClient(setting("API_BASE_URL"), setting("DATABASE_URL") or setting("SUPABASE_DB_URL"))
    return client.get(route, **params)


def frame_from(records):
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame.date, utc=True).dt.tz_convert("Europe/Istanbul")
    return frame.sort_values("date")


def plot_consumption(frame, daily=False):
    values = frame.set_index("date")[["actual", "prediction", "epias_forecast"]].apply(pd.to_numeric)
    if daily:
        # An incomplete day must not appear as an artificially low daily total.
        values = values.resample("D").sum(min_count=24) / 1000
    else:
        values = values.reindex(pd.date_range(values.index.min(), values.index.max(), freq="h"))
    fig = go.Figure()
    for key, name, color, dash in [
        ("actual", "Actual consumption", "#1d3553", "solid"),
        ("prediction", "XGBoost", "#07897e", "solid"),
        ("epias_forecast", "EPIAS", "#d39436", "dot"),
    ]:
        if values[key].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values[key],
                    name=name,
                    mode="lines+markers" if len(values) <= 7 else "lines",
                    line={"color": color, "width": 2.5, "dash": dash},
                    connectgaps=False,
                )
            )
    fig.update_layout(
        height=410,
        margin={"l": 12, "r": 12, "t": 25, "b": 10},
        template="plotly_white",
        legend={"orientation": "h", "y": 1.12},
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title="Daily consumption · GWh" if daily else "Hourly consumption · MWh",
    )
    st.plotly_chart(fig, use_container_width=True)


def metric_cards(metrics):
    # Head-to-head cards use the same rows for BOTH forecasts.
    model = metrics["paired_model"] or metrics["model"]
    epias = metrics["epias"]
    columns = st.columns(3)
    for column, key, label, unit in zip(
        columns, ["mae_mwh", "rmse_mwh", "mape_pct"], ["MAE", "RMSE", "MAPE"], ["MWh", "MWh", "%"]
    ):
        value = model[key] if model else None
        delta = None
        if value is not None and epias and epias[key] is not None:
            suffix = "pp" if key == "mape_pct" else "MWh"
            delta = f"{value - epias[key]:+,.2f} {suffix} vs EPIAS"
        column.metric(
            label, f"{value:,.2f} {unit}" if value is not None else "—", delta=delta, delta_color="inverse"
        )
        if epias and epias[key] is not None:
            column.caption(f"EPIAS: {epias[key]:,.2f} {unit}")
    hours = metrics["comparison_hours"] if epias else metrics["observed_hours"]
    st.caption(
        f"{hours:,} evaluated hours · {metrics['forecast_hours']:,} / "
        f"{metrics['expected_hours']:,} forecast hours available · Lower error is better."
    )
    if model and model["mape_hours"] < model["hours"]:
        st.caption("MAPE excludes zero-consumption hours; MAE and RMSE include them.")


st.title("Türkiye Electricity Forecast")
st.caption(
    "National electricity demand · XGBoost predictions compared with EPIAS forecasts and actual consumption"
)

try:
    status = fetch("/status")
except (requests.RequestException, ValueError) as error:
    explanations = {
        "authentication": "Supabase rejected the database credentials. Check the username and password in SUPABASE_DB_URL.",
        "dns": "The Supabase database hostname could not be resolved from this computer.",
        "network_route": "This computer cannot reach the Supabase database address. Use the Supabase session-pooler connection string for local IPv4 access.",
        "timeout": "The Supabase database connection timed out. Check the network and database availability.",
        "tls": "The secure Supabase database connection could not be established.",
    }
    reason = explanations.get(str(error), "The configured database could not be reached from this computer.")
    st.error(reason)
    st.caption("The connection string is not displayed or logged.")
    if st.button("Retry"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

if not status["latest_target_date"]:
    st.info("No monitoring records are available in the connected database.")
    st.stop()

source = "all"
first = date.fromisoformat(status["first_target_date"])
last = date.fromisoformat(status["latest_target_date"])
whole_history = (first, last)

info_column, refresh_column = st.columns([6, 1])
info_column.caption(f"Data available: {first:%d %b %Y} — {last:%d %b %Y} · Istanbul time (UTC+03)")
if refresh_column.button("Refresh", use_container_width=True):
    st.session_state["history_range"] = whole_history
    st.cache_data.clear()
    st.rerun()
try:
    dates = st.date_input(
        "Date range",
        whole_history,
        min_value=first,
        max_value=last,
        key="history_range",
    )
    if len(dates) != 2:
        st.info("Choose an end date to view the selected period.")
    else:
        start, end = dates
        records = fetch("/forecasts", start=str(start), end=str(end), source=source)["records"]
        if not records:
            st.info("No records in the selected date range.")
        else:
            metrics = evaluate_records(records, ((end - start).days + 1) * 24)
            metric_cards(metrics)
            frame = frame_from(records)
            st.subheader("Consumption over time")
            plot_consumption(frame, daily=True)
            with st.expander("Monthly performance"):
                rows = []
                for month, subset in frame.groupby(frame.date.dt.strftime("%Y-%m")):
                    result = evaluate_records(subset.to_dict("records"))
                    model = result["paired_model"] or result["model"]
                    epias = result["epias"]
                    if model:
                        rows.append(
                            {
                                "Month": month,
                                "XGBoost MAE (MWh)": model["mae_mwh"],
                                "EPIAS MAE (MWh)": epias["mae_mwh"] if epias else None,
                                "XGBoost MAPE (%)": model["mape_pct"],
                                "EPIAS MAPE (%)": epias["mape_pct"] if epias else None,
                                "Evaluated hours": model["hours"],
                            }
                        )
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            with st.expander("Hourly records"):
                columns = ["date", "actual", "prediction", "epias_forecast"]
                st.dataframe(frame[columns], hide_index=True, use_container_width=True)
                st.download_button(
                    "Download CSV",
                    frame[columns].to_csv(index=False),
                    "electricity-demand.csv",
                    "text/csv",
                )
except (requests.RequestException, ValueError, KeyError):
    st.error("The selected data could not be loaded. Please refresh and try again.")
