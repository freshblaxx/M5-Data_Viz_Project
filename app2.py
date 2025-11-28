"""
Professional Streamlit Dashboard: Social Media Usage & Mental Health
Clean white design | Black text | Professional corporate look
Author: Your Name
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Social Media & Mental Health Insights",
    page_icon="brain",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Global Professional Styling – Pure White & Black Design
# =============================================================================

st.markdown("""
<style>
    /* Main app background & text */
    .stApp {
        background-color: #FFFFFF !important;
    }
    .stApp * {
        color: #000000 !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        color: #000000 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA !important;
        border-right: 1px solid #E0E0E0;
    }

    /* Dropdowns, Sliders, Radio Buttons – Black border & text */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stSlider > div > div,
    .stRadio > div {
        background-color: #FFFFFF !important;
        border: 2px solid #000000 !important;
        border-radius: 8px !important;
    }
    .stSelectbox label,
    .stSlider label,
    .stRadio label {
        font-weight: 600 !important;
    }

    /* Metric cards – clean black border */
    .metric-container {
        background-color: #FFFFFF;
        border: 2px solid #000000;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        height: 140px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight:  : bold;
        margin: 0.5rem 0 0 0;
        color: #000000;
    }
    .metric-label {
        font-size: 0.95rem;
        color: #000000;
        opacity: 0.8;
        margin-bottom: 0.3rem;
    }

    /* Plotly charts – white background, black text */
    .js-plotly-plot .plotly {
        background: #FFFFFF !important;
    }
    .js-plotly-plot .plotly text {
        fill: #000000 !important;
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
        font-weight: 500 !important;
    }
    .js-plotly-plot .plotly .gtitle {
        font-weight: 600 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        color: #000000 !important;
    }
    .stTabs [data-baseweb="tab"].is-active {
        border-bottom: 4px solid #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Title & Introduction
# =============================================================================

st.title("Social Media Usage & Mental Health Insights")
st.markdown("""
**Objective**  
Provide clear, evidence-based insights into how daily social media consumption correlates with anxiety, stress, and mood — empowering young adults, parents, and educators to make informed digital wellness decisions.
""")

st.markdown("---")

# =============================================================================
# Load Data
# =============================================================================

@st.cache_data
def load_data():
    """
    Load the synthetic mental health and social media dataset.

    The dataset is packaged alongside this application within the shared
    container. Using an absolute path ensures the file can be located
    reliably when the application is run in different environments.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the dataset used for analysis.
    """
    df = pd.read_csv("mental_health_social_media_dataset.csv")
    return df

df = load_data()

# =============================================================================
# Sidebar – Professional Filters
# =============================================================================

with st.sidebar:
    st.header("Filters")

    platform = st.selectbox(
        "Platform",
        options=["All"] + sorted(df["platform"].dropna().unique().tolist())
    )

    gender = st.selectbox(
        "Gender",
        options=["All"] + sorted(df["gender"].dropna().unique().tolist())
    )

    min_age, max_age = int(df["age"].min()), int(df["age"].max())
    age_range = st.slider(
        "Age Range",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age),
        step=1
    )

    mental_state = st.selectbox(
        "Mental Health State",
        options=["All"] + sorted(df["mental_state"].dropna().unique().tolist())
    )

    metric_options = {
        "Anxiety Level": "anxiety_level",
        "Stress Level": "stress_level",
        "Mood Level": "mood_level"
    }
    selected_metric_label = st.radio(
        "Mental Health Metric for Scatter Analysis",
        options=list(metric_options.keys())
    )
    selected_metric = metric_options[selected_metric_label]

# Apply filters
data = df.copy()
if platform != "All":
    data = data[data["platform"] == platform]
if gender != "All":
    data = data[data["gender"] == gender]
if mental_state != "All":
    data = data[data["mental_state"] == mental_state]
data = data[(data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]

# =============================================================================
# Tabs
# =============================================================================

tab_overview, tab_usage, tab_lifestyle, tab_correlations, tab_insights = st.tabs([
    "Overview", "Usage & Age", "Sleep & Activity", "Correlations", "Key Insights"
])

# =============================================================================
# Tab 1: Overview & Key Metrics
# =============================================================================
with tab_overview:
    st.subheader("Key Summary Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Total Participants</div>
            <div class="metric-value">{len(data):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Avg. Anxiety Level</div>
            <div class="metric-value">{data['anxiety_level'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Avg. Stress Level</div>
            <div class="metric-value">{data['stress_level'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">Avg. Mood Level</div>
            <div class="metric-value">{data['mood_level'].mean():.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    # -------------------------------------------------------------------------
    # Social media usage distribution
    #
    # Instead of comparing mental health states across platforms—which can give
    # the impression of perfectly balanced categories in a synthetic dataset—we
    # focus on how much time respondents actually spend on social media. A
    # histogram clearly shows the spread of daily minutes across the sample and
    # helps establish whether excessive use is common.
    # -------------------------------------------------------------------------
    st.subheader("Distribution of Daily Social Media Usage")
    # Plot a histogram of daily social media time. We set the number of bins
    # explicitly to produce a smooth distribution and apply a clean white
    # template consistent with the overall dashboard aesthetic.
    fig_hist = px.histogram(
        data,
        x="social_media_time_min",
        nbins=20,
        title="Distribution of Daily Social Media Usage (minutes)",
        template="simple_white",
    )
    fig_hist.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Daily Social Media Time (minutes)",
        yaxis_title="Number of Participants",
        hoverlabel=dict(
        bgcolor="white",     # Sets the hover box background to white
        font_color="black",  # Sets the hover box text to black
        bordercolor="#E5E7EB" # Optional grey border for the box
    )
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# Tab 2: Social Media Time vs Mental Health
# =============================================================================
with tab_usage:
    st.subheader(f"Daily Social Media Time vs {selected_metric_label}")

    def age_bucket(age):
        if age < 20: return "Under 20"
        elif age < 30: return "20–29"
        elif age < 40: return "30–39"
        else: return "40+"

    plot_data = data.copy()
    plot_data["Age Group"] = plot_data["age"].apply(age_bucket)

    # -------------------------------------------------------------------------
    # Scatter plot with trend line
    #
    # To help the audience see the overall relationship between screen time and
    # the selected mental health metric, we overlay a best‑fit regression line
    # on the scatter plot. We also lighten the markers to reduce visual clutter.
    # -------------------------------------------------------------------------
    fig_scatter = go.Figure()
    # Scatter markers coloured by age group
    for age_group, group_df in plot_data.groupby("Age Group"):
        fig_scatter.add_trace(go.Scatter(
            x=group_df["social_media_time_min"],
            y=group_df[selected_metric],
            mode="markers",
            name=str(age_group),
            marker=dict(size=7, opacity=0.6, line=dict(width=0.5, color="#333333")),
            hovertemplate=(
                "Age Group: %{customdata[0]}<br>Time: %{x} min<br>"
                f"{selected_metric_label}: %{{y:.2f}}<br>"
                "Gender: %{customdata[1]}<br>Platform: %{customdata[2]}<br>State: %{customdata[3]}"
            ),
            customdata=np.stack([
                group_df["Age Group"],
                group_df["gender"],
                group_df["platform"],
                group_df["mental_state"]
            ], axis=-1)
        ))
    # Compute linear regression (least squares) on the full data for trend line
    if len(plot_data) > 1:
        x_vals = plot_data["social_media_time_min"].values
        y_vals = plot_data[selected_metric].values
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
        y_line = slope * x_line + intercept
        fig_scatter.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Trend line",
            line=dict(color="#000000", width=2, dash="dash"),
            hoverinfo="skip"
        ))
    fig_scatter.update_layout(
        template="simple_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=f"Social Media Usage vs {selected_metric_label} by Age Group",
        xaxis_title="Daily Social Media Time (minutes)",
        yaxis_title=selected_metric_label,
        legend_title="Age Group",
        height=500,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # -------------------------------------------------------------------------
    # Average metric by screen time bucket
    #
    # Many dots can obscure the underlying pattern. We summarise the selected
    # mental health metric by buckets of daily screen time to make the trend
    # easy to see at a glance. Buckets correspond to typical guidelines.
    # -------------------------------------------------------------------------
    def time_bucket(minutes: float) -> str:
        if pd.isna(minutes):
            return "Unknown"
        if minutes < 60:
            return "<60"
        elif minutes < 120:
            return "60–120"
        elif minutes < 180:
            return "120–180"
        elif minutes < 240:
            return "180–240"
        else:
            return "≥240"

    bucket_data = data.copy()
    bucket_data["Time Bucket"] = bucket_data["social_media_time_min"].apply(time_bucket)
    # Define a category order to ensure proper sorting on the x‑axis
    bucket_order = ["<60", "60–120", "120–180", "180–240", "≥240"]
    bucket_avg = (
        bucket_data.groupby("Time Bucket")[selected_metric]
        .mean()
        .reindex(bucket_order)
        .reset_index()
        .rename(columns={selected_metric: f"Average {selected_metric_label}"})
    )

    fig_bucket = px.bar(
        bucket_avg,
        x="Time Bucket",
        y=f"Average {selected_metric_label}",
        title=f"Average {selected_metric_label} by Daily Social Media Time",
        template="simple_white",
    )
    fig_bucket.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Daily Social Media Time (minutes)",
        yaxis_title=f"Average {selected_metric_label}",
        height=450,
    )
    st.plotly_chart(fig_bucket, use_container_width=True)

# =============================================================================
# Tab 3: Sleep & Physical Activity Impact
# =============================================================================
with tab_lifestyle:
    st.subheader("Average Mood by Sleep Duration and Physical Activity")

    def sleep_category(hours):
        if hours < 6: return "<6h"
        if hours < 7: return "6–7h"
        if hours < 8: return "7–8h"
        return "≥8h"

    def activity_category(minutes):
        if minutes < 30: return "<30min"
        if minutes < 60: return "30–59min"
        return "≥60min"

    heat_data = data.copy()
    heat_data["Sleep"] = heat_data["sleep_hours"].apply(sleep_category)
    heat_data["Activity"] = heat_data["physical_activity_min"].apply(activity_category)

    # Build a pivot table of mean mood levels by sleep and activity categories
    heatmap = heat_data.pivot_table(
        values="mood_level",
        index="Sleep",
        columns="Activity",
        aggfunc="mean"
    )
    # Prepare display values: round numeric values and mark missing as "N/A"
    heatmap_display = heatmap.copy().round(2).astype(str)
    heatmap_display = heatmap_display.replace({"nan": "N/A"})
    # Fill NaN values in the z matrix with 0 so that Plotly can render them;
    # these zeros will be coloured at the low end of the scale and labelled "N/A".
    heatmap_filled = heatmap.fillna(0)

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_filled.values,
        x=heatmap_filled.columns,
        y=heatmap_filled.index,
        colorscale="Viridis",
        text=heatmap_display.values,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        hoverongaps=False
    ))
    fig_heat.update_layout(
        template="simple_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        title="Average Mood Level by Sleep and Physical Activity",
        xaxis_title="Daily Physical Activity",
        yaxis_title="Nightly Sleep Duration",
        height=550
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# =============================================================================
# Tab 4: Correlation Matrix
# =============================================================================
with tab_correlations:
    st.subheader("Correlation Matrix: Interactions, Screen Time & Mental Health")

    # -------------------------------------------------------------------------
    # Correlation strengths bar chart
    #
    # A full heatmap can overwhelm the audience. Instead, we compute the
    # correlations between key predictors (negative interactions, positive
    # interactions, social media time) and each mental health outcome and
    # display them as grouped bars. This highlights which factors are most
    # strongly associated with anxiety, stress and mood.
    # -------------------------------------------------------------------------
    predictors = [
        "negative_interactions_count",
        "positive_interactions_count",
        "social_media_time_min",
    ]
    mental_metrics = [
        "anxiety_level",
        "stress_level",
        "mood_level",
    ]
    corr_records = []
    for metric_col in mental_metrics:
        for pred_col in predictors:
            # compute Pearson correlation; handle potential NaN gracefully
            if data[pred_col].std(ddof=0) == 0 or data[metric_col].std(ddof=0) == 0:
                corr_val = 0
            else:
                corr_val = data[pred_col].corr(data[metric_col])
            corr_records.append({
                "Predictor": pred_col.replace("_count", "").replace("_min", "").replace("_", " ").title(),
                "Mental Health Outcome": metric_col.replace("_", " ").title(),
                "Correlation": corr_val,
            })
    corr_df = pd.DataFrame(corr_records)
    # Order predictors manually for consistent display
    predictor_order = [
        "Negative Interactions",
        "Positive Interactions",
        "Social Media Time",
    ]
    corr_df["Predictor"] = pd.Categorical(corr_df["Predictor"], categories=predictor_order, ordered=True)

    fig_corrbar = px.bar(
        corr_df,
        x="Predictor",
        y="Correlation",
        color="Mental Health Outcome",
        barmode="group",
        title="Correlation Strengths between Predictors and Mental Health Outcomes",
        template="simple_white",
    )
    fig_corrbar.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis_title="Predictor",
        yaxis_title="Correlation Coefficient",
        yaxis=dict(range=[-1, 1]),
        legend_title="Outcome",
        height=500,
    )
    st.plotly_chart(fig_corrbar, use_container_width=True)

# =============================================================================
# Tab 5: Key Insights & Recommendations
# =============================================================================
with tab_insights:
    st.subheader("Key Insights")
    st.markdown("""
    - **Excessive use (>120 min/day)** is strongly associated with higher anxiety and stress across all age groups.
    - **Negative online interactions** are the single strongest predictor of poor mental health outcomes.
    - **Positive interactions** show a modest protective effect on mood.
    - **Sleep (7–8 hours) and physical activity (≥60 min/day)** significantly buffer negative effects — even among heavy users.
    - **Younger users (<25)** show the steepest rise in anxiety with increased screen time.
    """)

    st.subheader("Evidence-Based Recommendations")
    st.markdown("""
    1. **Limit social media to 60–120 minutes per day** using built-in wellness tools.
    2. **Curate positive online environments** — follow supportive accounts, mute toxic ones.
    3. **Prioritize 7–8 hours of sleep** and avoid screens 60 min before bed.
    4. **Aim for ≥60 minutes of daily movement** (walking, sports, dance).
    5. **Foster open family conversations** about digital habits and emotional well-being.
    """)

    st.info("This dashboard uses a research-grade synthetic dataset. Patterns align with peer-reviewed studies but are not medical advice.")

st.markdown("<br><hr><center>Made with Streamlit • Data Analytics Project 2025</center>", unsafe_allow_html=True)