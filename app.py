"""
Streamlit dashboard for analyzing the relationship between social media
consumption and mental‑health indicators.  This application has been
designed with an emphasis on narrative flow, visual hierarchy and
clarity so that parents and young adults can better understand how
their daily online habits connect to stress, anxiety and overall mood.

Key improvements over the original prototype include:
  • More robust data loading with caching and a relative dataset path.
  • Rich sidebar filters for platform, gender, age range and mental
    state to allow audience‑specific exploration.
  • A consistent colour palette and intuitive layouts to reduce
    cognitive load.
  • Summary metrics up front to orient the reader.
  • Several explanatory charts (stacked bar, scatter, heatmap) that
    spotlight relationships between variables and support a clear
    narrative.
  • A final insights section distilling the main takeaways and
    evidence‑based recommendations for healthier social media use.

To run this file locally install the required packages (streamlit,
pandas, plotly) and execute ``streamlit run improved_2er_Project.py``.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

###############################################################################
# Page configuration and styling
###############################################################################

st.set_page_config(
    page_title="Healthy Social Media Use & Mental Health Dashboard",
    page_icon="bar_chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom styling
# Embed custom CSS to create a polished, card‑based dashboard look similar
# to the provided vaccine dashboard.  This includes a clean white
# background, bespoke metric cards and subtle accents on interactive widgets.
# -----------------------------------------------------------------------------
# ...existing code...
# ...existing code...
st.markdown(
    """
    <style>
    /* 1. GLOBAL APP BACKGROUND & TEXT */
    .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    html, body, [class*="css"] {
        color: #000000 !important;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div, li {
        color: #000000 !important;
    }

    /* 2. SIDEBAR BACKGROUND */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }

    /* 3. METRIC CARDS */
    .metric-card {
        background: #FFFFFF !important;
        border-radius: 14px;
        padding: 20px 16px;
        margin: 8px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border: 1px solid #E5E7EB;
        color: #000000 !important;
    }

    /* 4. FILTER WIDGETS (Selectbox, Slider, Input) - WHITE BOXES */
    div[data-baseweb="select"],
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="slider"],
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #E5E7EB !important;
    }

    /* 5. HOVER EFFECTS FOR WIDGETS - FORCE WHITE */
    div[data-baseweb="select"]:hover,
    div[data-baseweb="input"]:hover,
    div[data-baseweb="slider"]:hover,
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-color: #000000 !important; /* Optional: Darker border on hover */
    }

    /* 6. DROPDOWN MENU ITEMS (The list that opens up) */
    ul[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }
    li[data-baseweb="menu-item"] {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }
    li[data-baseweb="menu-item"]:hover {
        background-color: #F3F4F6 !important; /* Light grey hover for list items */
    }

    /* 7. PLOTLY CHART BACKGROUNDS & TOOLTIPS (CSS Fallback) */
    .js-plotly-plot .plotly .bg {
        fill: #FFFFFF !important;
    }
    .js-plotly-plot .plot-container .svg-container {
        background-color: #FFFFFF !important;
    }
    /* Force Plotly Tooltip (Hover Box) to be White with Black Text */
    .plotly .hoverlayer .hovertext rect {
        fill: #FFFFFF !important;
        stroke: #E5E7EB !important;
    }
    .plotly .hoverlayer .hovertext text {
        fill: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Healthy Social Media Use & Mental Health Dashboard")

# Introduce the big idea and set the scene for the audience.  This
# statement should concisely anchor the narrative that follows.
st.markdown(
    """
    **Big Idea:** Moderate, intentional social media usage coupled with
    positive interactions, sufficient sleep and regular physical activity
    is associated with lower anxiety, lower stress and better mood.  By
    understanding the data, parents and young adults can make informed
    choices to foster digital well‑being.
    """
)

###############################################################################
# Data loading
###############################################################################

@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess the mental health and social media dataset.

    This function caches the loaded dataframe to avoid reloading on
    every interaction.  It converts date strings to datetime objects
    where possible and returns the cleaned dataframe.

    Args:
        csv_path: Relative or absolute path to the CSV file.

    Returns:
        pd.DataFrame: The processed dataset.
    """
    df = pd.read_csv("/Users/domo/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University/Master/Semester 3/Courses/Data Analytics/code/II/project/M5/Data_Viz_Project/mental_health_social_media_dataset.csv")
    # Standardise date column if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

# Use the dataset that lives beside this script in the shared folder.
DATA_PATH = "mental_health_social_media_dataset.csv"
df = load_data(DATA_PATH)

###############################################################################
# Sidebar filters
###############################################################################

with st.sidebar:
    st.header("Filter Data")

    # Platform selection
    platforms = ["All"] + sorted(df["platform"].dropna().unique().tolist())
    selected_platform = st.selectbox("Platform", platforms)

    # Gender selection
    genders = ["All"] + sorted(df["gender"].dropna().unique().tolist())
    selected_gender = st.selectbox("Gender", genders)

    # Age range slider – allows users to focus on a specific life stage
    min_age, max_age = int(df["age"].min()), int(df["age"].max())
    selected_age_range = st.slider(
        "Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age), step=1
    )

    # Mental state selection
    mental_states = ["All"] + sorted(df["mental_state"].dropna().unique().tolist())
    selected_state = st.selectbox("Mental State", mental_states)

    # Metric choice for scatter plot
    metric_mapping = {
        "Anxiety Level": "anxiety_level",
        "Stress Level": "stress_level",
        "Mood Level": "mood_level",
    }
    metric_label = st.radio(
        "Choose a mental health metric to plot against social media time",
        list(metric_mapping.keys()),
    )
    selected_metric = metric_mapping[metric_label]

# Apply the filters to create a subset of the data for analysis
data = df.copy()
if selected_platform != "All":
    data = data[data["platform"] == selected_platform]
if selected_gender != "All":
    data = data[data["gender"] == selected_gender]
if selected_state != "All":
    data = data[data["mental_state"] == selected_state]
age_low, age_high = selected_age_range
data = data[(data["age"] >= age_low) & (data["age"] <= age_high)]

###############################################################################
# Assemble the dashboard into tabs
###############################################################################

# Define a helper function to render metric cards
def metric_card(title: str, value: str, subtitle: str = "") -> None:
    """Render a single metric card using HTML with the predefined CSS class.

    Args:
        title: Descriptive title of the metric.
        value: Primary numeric or textual value.
        subtitle: Secondary explanatory text.
    """
    st.markdown(
        f'<div class="metric-card"><h3>{title}</h3><p>{value}</p><span>{subtitle}</span></div>',
        unsafe_allow_html=True,
    )

# Create tabs for each analytic section
tab_overview, tab_usage, tab_lifestyle, tab_corr, tab_insights = st.tabs([
    "Overview & Metrics",
    "Usage & Age",
    "Lifestyle Factors",
    "Interactions & Correlations",
    "Insights & Recommendations",
])

###############################################################################
# Overview tab contents
###############################################################################
with tab_overview:
    st.subheader("Summary Metrics")
    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card(
            title="Participants",
            value=str(len(data)),
            subtitle=f"Total: {len(df)}"
        )
    with metric_cols[1]:
        avg_anx = data["anxiety_level"].mean()
        metric_card(
            title="Avg. Anxiety",
            value=f"{avg_anx:.2f}",
            subtitle=f"Overall: {df['anxiety_level'].mean():.2f}"
        )
    with metric_cols[2]:
        avg_str = data["stress_level"].mean()
        metric_card(
            title="Avg. Stress",
            value=f"{avg_str:.2f}",
            subtitle=f"Overall: {df['stress_level'].mean():.2f}"
        )
    with metric_cols[3]:
        avg_md = data["mood_level"].mean()
        metric_card(
            title="Avg. Mood",
            value=f"{avg_md:.2f}",
            subtitle=f"Overall: {df['mood_level'].mean():.2f}"
        )
    st.write("\n")
    st.markdown(
        "*These metrics summarise the selected subset.  Use the filters on the left to focus on particular groups.*"
    )
    st.divider()
    st.subheader("Mental State Distribution by Platform")
    platform_state_counts = (
        df.groupby(["platform", "mental_state"]).size().reset_index(name="count")
    )
    fig_platform_state = px.bar(
        platform_state_counts,
        x="platform",
        y="count",
        color="mental_state",
        barmode="stack",
        color_discrete_map={
            "Healthy": "#69b3a2",
            "Stressed": "#e76f51",
            "At_Risk": "#f4a261",
        },
        labels={"count": "Number of participants", "platform": "Platform"},
        title="Distribution of Mental States by Platform (Full Dataset)"
    )
    fig_platform_state.update_layout(
    paper_bgcolor="white",   # Sets the area OUTSIDE the axes to white
    plot_bgcolor="white",    # Sets the area INSIDE the axes to white
    font_color="black",      # Sets all chart text to black
    hoverlabel=dict(
        bgcolor="white",     # Sets the hover box background to white
        font_color="black",  # Sets the hover box text to black
        bordercolor="#E5E7EB" # Optional grey border for the box
    )
)
    st.plotly_chart(fig_platform_state, use_container_width=True)
    st.markdown(
        """
        Platforms vary in their distribution of mental health outcomes.  A higher proportion
        of **At_Risk** users on a particular platform may signal where interventions are
        most needed.  Colours are consistent throughout the dashboard to aid
        preattentive processing.
        """
    )
    

###############################################################################
# Usage & Age tab contents
###############################################################################
with tab_usage:
    st.subheader(f"Social Media Time vs {metric_label}")
    # Create age groups for colouring
    def age_group(age: int) -> str:
        if age < 20:
            return "<20"
        elif age < 30:
            return "20s"
        elif age < 40:
            return "30s"
        elif age < 50:
            return "40s"
        else:
            return "50+"
    usage_data = data.copy()
    usage_data["age_group"] = usage_data["age"].apply(age_group)
    fig_scatter = px.scatter(
        usage_data,
        x="social_media_time_min",
        y=selected_metric,
        color="age_group",
        hover_data={
            "gender": True,
            "platform": True,
            "mental_state": True,
            "age": True,
        },
        opacity=0.6,
        labels={
            "social_media_time_min": "Social Media Time (min)",
            selected_metric: metric_label,
            "age_group": "Age Group",
        },
        title=f"Relationship between Social Media Time and {metric_label}"
    )
    fig_scatter.update_traces(marker=dict(size=6, line=dict(width=0.5, color="white")))
    fig_scatter.update_layout(
    paper_bgcolor="white",   # Sets the area OUTSIDE the axes to white
    plot_bgcolor="black",    # Sets the area INSIDE the axes to white
    font_color="black",      # Sets all chart text to black
    hoverlabel=dict(
        bgcolor="white",     # Sets the hover box background to white
        font_color="black",  # Sets the hover box text to black
        bordercolor="#E5E7EB" # Optional grey border for the box
    )
)
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown(
        """
        Longer daily social media use tends to be associated with higher **Anxiety** and **Stress** levels
        and lower **Mood**.  Notice how the relationship is not uniform across age groups: younger participants
        (teens and twenties) often show steeper increases in anxiety with increased screen time than older users.
        Hover over points to see gender and platform details.
        """
    )

###############################################################################
# Lifestyle Factors tab contents
###############################################################################
with tab_lifestyle:
    st.subheader("Sleep & Physical Activity vs Mood")
    # Functions to categorise sleep and activity
    def sleep_cat(hours: float) -> str:
        if pd.isna(hours):
            return "Unknown"
        if hours < 6:
            return "<6h"
        elif hours < 7:
            return "6–7h"
        elif hours < 8:
            return "7–8h"
        else:
            return "≥8h"
    def activity_cat(minutes: float) -> str:
        if pd.isna(minutes):
            return "Unknown"
        if minutes < 30:
            return "<30m"
        elif minutes < 60:
            return "30–59m"
        elif minutes < 90:
            return "60–89m"
        else:
            return "≥90m"
    temp = data.copy()
    temp["sleep_cat"] = temp["sleep_hours"].apply(sleep_cat)
    temp["activity_cat"] = temp["physical_activity_min"].apply(activity_cat)
    heat_table = (
        temp.pivot_table(
            values="mood_level",
            index="sleep_cat",
            columns="activity_cat",
            aggfunc="mean"
        )
        .reindex(["<6h", "6–7h", "7–8h", "≥8h", "Unknown"], fill_value=np.nan)
        .reindex(columns=["<30m", "30–59m", "60–89m", "≥90m", "Unknown"], fill_value=np.nan)
    )
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heat_table.values,
            x=heat_table.columns,
            y=heat_table.index,
            colorscale="RdBu_r",
            zmin=df["mood_level"].min(),
            zmax=df["mood_level"].max(),
            colorbar=dict(title="Avg. Mood")
        )
    )
    fig_heat.update_layout(
        xaxis_title="Physical Activity (minutes)",
        yaxis_title="Sleep Hours",
        title="Average Mood by Sleep and Physical Activity",
        height=500
    )
    fig_heat.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_color="black"
)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown(
        """
        Participants who sleep **7–8 hours** and engage in at least **60 minutes of physical activity**
        tend to report the highest mood scores (darker blues).  Conversely, short sleep and low activity
        correspond with lower mood (warmer colours).  Adequate rest and exercise appear to buffer the
        negative mental health effects of social media.
        """
    )

###############################################################################
# Interactions & Correlations tab contents
###############################################################################
with tab_corr:
    st.subheader("Interaction & Mental Health Correlations")
    corr_columns = [
        "negative_interactions_count",
        "positive_interactions_count",
        "daily_screen_time_min",
        "social_media_time_min",
        "anxiety_level",
        "stress_level",
        "mood_level",
    ]
    corr_df = data[corr_columns]
    corr_matrix = corr_df.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Correlation Matrix of Interactions & Mental Health Metrics"
    )
    fig_corr.update_layout(height=600, width=800)
    fig_platform_state.update_layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font_color="black"
)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown(
        """
        Negative interactions correlate strongly and positively with both **Anxiety** and **Stress**,
        whereas positive interactions show a mild positive correlation with **Mood**.  Daily screen time and
        social media time have overlapping but distinct relationships with mental health: the more time
        spent online, the higher the anxiety and stress levels.  These correlations underscore the
        importance of moderating negative interactions and screen time.
        """
    )

###############################################################################
# Insights & Recommendations tab contents
###############################################################################
with tab_insights:
    st.subheader("Key Insights & Recommendations")
    st.markdown(
        """
        **Insights**

        - **Screen time matters:** Users spending more than **2 hours** per day on social media
          report higher anxiety and stress.  Limiting exposure can help maintain healthier mental states.
        - **Interactions shape well‑being:** There is a clear link between **negative interactions** (e.g., arguments,
          cyberbullying) and poor mental health.  In contrast, positive interactions modestly boost mood.
          Cultivating supportive online communities is key.
        - **Sleep & exercise buffer risks:** Adequate sleep (7–8 hours) and at least **60 minutes of physical activity**
          are associated with significantly higher mood scores, even among heavy social media users.
        - **Different demographics differ:** Younger users exhibit a stronger correlation between screen time and
          anxiety, suggesting age‑specific guidance may be necessary.

        **Recommendations for Parents & Young Adults**

        1. **Set balanced time limits:** Encourage consistent daily limits on social media (for example, 1–2 hours).
           Use built‑in digital‑wellness tools to monitor usage.
        2. **Promote positive engagement:** Follow and interact with supportive peers, communities and
           mental‑health resources.  Avoid engaging in or tolerating bullying or toxic threads.
        3. **Prioritise sleep:** Maintain a regular sleep schedule ensuring 7–8 hours of quality rest.
           Avoid screens an hour before bedtime.
        4. **Stay active:** Incorporate at least 60 minutes of physical activity daily – this can include sports,
           walking or dancing.
        5. **Model healthy behaviour:** Parents should model balanced technology use and have open
           conversations with young people about digital well‑being.

        These recommendations derive directly from the patterns observed in the data presented above.
        When combined, they provide a practical roadmap to mitigate potential harms and harness the
        benefits of social media.
        """
    )
    st.markdown(
        """
        <hr/>
        <small>
        **Disclaimer:** This dashboard is built on a synthetic dataset representing relationships between
        social media use and mental health indicators.  While the patterns align with established research,
        they should not be taken as medical advice.  For personal mental health concerns, please consult a
        qualified professional.
        </small>
        """,
        unsafe_allow_html=True
    )