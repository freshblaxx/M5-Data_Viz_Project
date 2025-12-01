import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Digital Environments & Mental Well-Being",
    layout="wide"
)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load dataset from the same folder as this script."""
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "/Users/domo/Library/Mobile Documents/com~apple~CloudDocs/Desktop/University/Master/Semester 3/Courses/Data Analytics/code/II/project/M5/Data_Viz_Project/mental_health_social_media_dataset.csv")
    df = pd.read_csv(file_path)

    # Age groups for colour in scatter plot
    def age_group(age):
        if age < 18:
            return "Under 18"
        elif age < 25:
            return "18–24"
        elif age < 35:
            return "25–34"
        else:
            return "35+"

    df["age_group"] = df["age"].apply(age_group)
    return df


df = load_data()


# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.title("Filter Data")

gender_opts = ["All"] + sorted(df["gender"].dropna().unique().tolist())
gender_sel = st.sidebar.selectbox("Gender", gender_opts, key="flt_gender")

platform_opts = ["All"] + sorted(df["platform"].dropna().unique().tolist())
platform_sel = st.sidebar.selectbox("Platform", platform_opts, key="flt_platform")

age_min, age_max = int(df["age"].min()), int(df["age"].max())
age_range = st.sidebar.slider(
    "Age Range", age_min, age_max, (age_min, age_max), key="flt_age_range"
)

mental_opts = ["All"] + sorted(df["mental_state"].dropna().unique().tolist())
mental_sel = st.sidebar.selectbox("Mental State", mental_opts, key="flt_mental")

# Apply filters to data
data = df.copy()
if gender_sel != "All":
    data = data[data["gender"] == gender_sel]
if platform_sel != "All":
    data = data[data["platform"] == platform_sel]
data = data[(data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]
if mental_sel != "All":
    data = data[data["mental_state"] == mental_sel]


# -----------------------------------------------------------------------------
# Layout: Title + objective
# -----------------------------------------------------------------------------
st.title("Digital Environments & Mental Well-Being")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Digital Landscape",
        "Screen Time & Emotional Load",
        "Interaction Quality",
        "Healthy Habits",
        "Key Learnings",
    ]
)


# -----------------------------------------------------------------------------
# Tab 1: Digital Landscape
# -----------------------------------------------------------------------------
with tab1:
    st.header("Who Is Online & How Much Time Do They Spend?")

    if len(data) == 0:
        st.warning("No participants match the selected filters. Try relaxing the filters in the sidebar.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        total_participants = len(data)
        avg_anxiety = data["anxiety_level"].mean()
        avg_stress = data["stress_level"].mean()
        avg_mood = data["mood_level"].mean()

        col1.metric("Total Participants", f"{total_participants}")
        col2.metric("Avg Anxiety (1–5)", f"{avg_anxiety:.2f}")
        col3.metric("Avg Stress (1–10)", f"{avg_stress:.2f}")
        col4.metric("Avg Mood (1–10)", f"{avg_mood:.2f}")

        st.markdown("### Daily Social Media Time (minutes)")
        hist = px.histogram(
            data,
            x="social_media_time_min",
            nbins=30,
            template="simple_white",
            labels={"social_media_time_min": "Daily social media time (minutes)", "count": "Participants"},
        )
        st.plotly_chart(hist, use_container_width=True)

        st.markdown("### Age Distribution")
        age_hist = px.histogram(
            data,
            x="age",
            nbins=20,
            template="simple_white",
            labels={"age": "Age", "count": "Participants"},
        )
        st.plotly_chart(age_hist, use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 2: Screen Time & Emotional Load
# -----------------------------------------------------------------------------
with tab2:
    st.header("Does More Screen Time Mean Worse Mental Health?")

    metric_choice = st.radio(
        "Choose the outcome you want to explore:",
        options=["Stress Level", "Anxiety Level", "Mood Level"],
        horizontal=True,
        key="screen_metric_choice",
    )
    metric_map = {
        "Stress Level": "stress_level",
        "Anxiety Level": "anxiety_level",
        "Mood Level": "mood_level",
    }
    selected_metric_col = metric_map[metric_choice]

    if len(data) < 2:
        st.warning("Not enough data points for this filter combination to show a relationship.")
    else:
        st.markdown("### Screen Time vs Selected Mental Health Metric")

        x_vals = data["social_media_time_min"].values
        y_vals = data[selected_metric_col].values

        # simple linear trend
        m, b = np.polyfit(x_vals, y_vals, 1)
        trend_y = m * x_vals + b

        scatter_fig = go.Figure()
        scatter_fig.add_trace(
            go.Scatter(
                x=data["social_media_time_min"],
                y=data[selected_metric_col],
                mode="markers",
                marker=dict(size=6, opacity=0.6),
                name=metric_choice,
                text=data["age_group"],
                hovertemplate=(
                    "Screen time: %{x} min<br>"
                    f"{metric_choice}: " + "%{y:.2f}<br>"
                    "Age group: %{text}<extra></extra>"
                ),
            )
        )
        scatter_fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=trend_y,
                mode="lines",
                name="Trend",
                line=dict(dash="dash"),
                hoverinfo="skip",
            )
        )
        scatter_fig.update_layout(
            template="simple_white",
            xaxis_title="Daily social media time (minutes)",
            yaxis_title=metric_choice,
            height=450,
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        st.markdown("### Average Scores by Screen-Time Bucket")

        def time_bucket(minutes):
            if minutes < 60:
                return "< 1h"
            elif minutes < 120:
                return "1–2h"
            elif minutes < 180:
                return "2–3h"
            else:
                return "3h+"

        bucket_data = data.copy()
        bucket_data["Time Bucket"] = bucket_data["social_media_time_min"].apply(time_bucket)
        bucket_order = ["< 1h", "1–2h", "2–3h", "3h+"]
        bucket_avg = (
            bucket_data.groupby("Time Bucket")[["anxiety_level", "stress_level", "mood_level"]]
            .mean()
            .reindex(bucket_order)
            .reset_index()
        )
        bucket_avg = bucket_avg.melt(
            id_vars="Time Bucket", var_name="Outcome", value_name="Average Value"
        )
        bucket_avg["Outcome"] = bucket_avg["Outcome"].str.replace("_", " ").str.title()

        bar = px.bar(
            bucket_avg,
            x="Time Bucket",
            y="Average Value",
            color="Outcome",
            barmode="group",
            template="simple_white",
            labels={"Average Value": "Average score"},
            title="Average Mental Health Scores by Screen-Time Bucket",
        )
        st.plotly_chart(bar, use_container_width=True)

        st.markdown(
            "_Heavier use is associated with higher stress and anxiety and slightly lower mood, "
            "but the size of the effect is moderate. To understand **how bad** things get, we also "
            "have to look at the quality of online interactions._"
        )


# -----------------------------------------------------------------------------
# Tab 3: Interaction Quality
# -----------------------------------------------------------------------------
with tab3:
    st.header("Interaction Quality Drives Mental Health")
    st.markdown(
        """
In this dataset, the **quality** of online interactions has a stronger association with mental health
than the **amount of time** spent online. Negative interactions (conflict, exclusion, toxicity) are
linked to higher stress and anxiety, while positive interactions provide only a modest lift to mood.
"""
    )

    if len(data) < 2:
        st.warning("Not enough data points to compute correlations for this filter combination.")
    else:
        predictors = [
            "negative_interactions_count",
            "positive_interactions_count",
            "social_media_time_min",
        ]
        # Order: Stress, Mood, Anxiety
        mental_metrics = ["stress_level", "mood_level", "anxiety_level"]

        corr_records = []
        for metric in mental_metrics:
            for pred in predictors:
                if data[pred].std() > 0 and data[metric].std() > 0:
                    corr_val = data[pred].corr(data[metric])
                else:
                    corr_val = 0.0
                corr_records.append(
                    {
                        "Predictor": pred.replace("_interactions_count", " interactions")
                                        .replace("_min", " time")
                                        .replace("_", " ")
                                        .title(),
                        "Outcome": metric.replace("_", " ").title(),
                        "Correlation": corr_val,
                    }
                )

        corr_df = pd.DataFrame(corr_records)
        corr_df["abs_corr"] = corr_df["Correlation"].abs()
        corr_df = corr_df.sort_values("abs_corr", ascending=True)

        corr_fig = go.Figure()
        for metric in mental_metrics:  # Stress, Mood, Anxiety
            outcome_name = metric.replace("_", " ").title()
            subset = corr_df[corr_df["Outcome"] == outcome_name]
            corr_fig.add_trace(
                go.Bar(
                    x=subset["Correlation"],
                    y=subset["Predictor"],
                    name=outcome_name,
                    orientation="h",
                    hovertemplate=(
                        f"Outcome: {outcome_name}<br>"
                        "Predictor: %{y}<br>"
                        "Correlation: %{x:.2f}<extra></extra>"
                    ),
                )
            )

        corr_fig.update_layout(
            template="simple_white",
            title="Correlation Strengths Between Predictors and Mental Health Outcomes",
            xaxis_title="Correlation coefficient",
            yaxis_title="Predictor",
            barmode="group",
            height=450,
        )
        st.plotly_chart(corr_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Impact of Negative Interactions on Mental Health")

        def neg_group(val: int) -> str:
            if val == 0:
                return "None"
            elif val == 1:
                return "A few"
            else:
                return "Many"

        neg_df = data.copy()
        neg_df["NegGroup"] = neg_df["negative_interactions_count"].apply(neg_group)

        neg_avg = (
            neg_df.groupby("NegGroup")[["anxiety_level", "stress_level", "mood_level"]]
            .mean()
            .reset_index()
        )

        neg_avg = neg_avg.melt(
            id_vars="NegGroup", var_name="Outcome", value_name="Average Value"
        )

        neg_avg["Outcome"] = neg_avg["Outcome"].str.replace("_", " ").str.title()
        neg_avg["OutcomeShort"] = neg_avg["Outcome"].str.replace(" Level", "")

        outcome_order = ["Stress", "Mood", "Anxiety"]
        group_order = ["None", "A few", "Many"]

        neg_avg["OutcomeGroup"] = neg_avg["OutcomeShort"] + " - " + neg_avg["NegGroup"]
        x_order = [f"{o} - {g}" for o in outcome_order for g in group_order]
        neg_avg["OutcomeGroup"] = pd.Categorical(
            neg_avg["OutcomeGroup"], categories=x_order, ordered=True
        )
        neg_avg = neg_avg.sort_values("OutcomeGroup")

        neg_avg["OutcomeShort"] = pd.Categorical(
            neg_avg["OutcomeShort"], categories=outcome_order, ordered=True
        )

        neg_fig = px.line(
            neg_avg,
            x="OutcomeGroup",
            y="Average Value",
            color="OutcomeShort",
            markers=True,
            template="simple_white",
            labels={
                "OutcomeGroup": "Outcome and negative interaction level",
                "Average Value": "Average score",
                "OutcomeShort": "Outcome",
            },
            title="Average Mental Health Outcomes by Negative Interaction Level",
        )
        neg_fig.update_layout(
            xaxis_tickangle=-35,
            height=450,
        )
        st.plotly_chart(neg_fig, use_container_width=True)

        st.markdown(
            "_Participants with many negative interactions report clearly higher stress and anxiety "
            "and lower mood. This suggests that hostile digital environments are more damaging than "
            "screen time alone._"
        )


# -----------------------------------------------------------------------------
# Tab 4: Healthy Habits – strongest positive predictors (clean version)
# -----------------------------------------------------------------------------
with tab4:
    st.header("Healthy Habits as Buffers")

    if len(data) < 2:
        st.warning("Not enough data points to analyse healthy habits for this filter combination.")
    else:
        # Candidate predictors
        predictors = [
            "positive_interactions_count",
            "sleep_hours",
            "physical_activity_min",
            "social_media_time_min",
            "age",
        ]

        outcomes = {
            "Mood Level": "mood_level",
            "Stress Level": "stress_level",
            "Anxiety Level": "anxiety_level",
        }

        # Correlations
        corr_rows = []
        for pred in predictors:
            if data[pred].std() == 0:
                continue
            for outcome_label, outcome_col in outcomes.items():
                if data[outcome_col].std() == 0:
                    continue
                corr_val = data[pred].corr(data[outcome_col])
                corr_rows.append(
                    {
                        "Predictor": pred,
                        "Outcome": outcome_label,
                        "Correlation": corr_val,
                    }
                )

        corr_df = pd.DataFrame(corr_rows)

        pred_labels = {
            "positive_interactions_count": "Positive Interactions",
            "sleep_hours": "Sleep Hours",
            "physical_activity_min": "Daily Physical Activity",
            "social_media_time_min": "Screen Time",
            "age": "Age",
        }
        corr_df["Predictor Label"] = corr_df["Predictor"].map(pred_labels)

        # Benefit score: + for mood, - for stress/anxiety
        corr_df["BenefitScore"] = corr_df.apply(
            lambda r: r["Correlation"] if r["Outcome"] == "Mood Level" else -r["Correlation"],
            axis=1,
        )

        # Top 3 positive predictors
        top_predictors = (
            corr_df.groupby("Predictor").BenefitScore.mean()
            .sort_values(ascending=False)
            .head(3)
            .index.tolist()
        )

        # Plot line charts for each predictor
        for pred in top_predictors:
            label = pred_labels[pred]
            st.subheader(f"{label} vs Mental Health")

            temp = data.copy()

            # Robust, meaningful intervals
            if pred == "sleep_hours":
                temp["Bucket"] = pd.cut(
                    temp[pred],
                    bins=[0, 7, 24],
                    labels=["<7h", "≥7h"],
                    include_lowest=True,
                )
            elif pred == "physical_activity_min":
                temp["Bucket"] = pd.cut(
                    temp[pred],
                    bins=[0, 30, 600],
                    labels=["<30m", "≥30m"],
                    include_lowest=True,
                )
            elif pred == "social_media_time_min":
                temp["Bucket"] = pd.cut(
                    temp[pred],
                    bins=[0, 60, 120, 180, temp[pred].max() + 0.1],
                    labels=["<1h", "1–2h", "2–3h", "3h+"],
                    include_lowest=True,
                )
            elif pred == "age":
                temp["Bucket"] = pd.cut(
                    temp[pred],
                    bins=[0, 18, 25, 35, temp[pred].max() + 0.1],
                    labels=["<18", "18–25", "25–35", "35+"],
                    include_lowest=True,
                )
            else:  # positive_interactions_count
                temp["Bucket"] = pd.cut(
                    temp[pred],
                    bins=[-1, 1, 3, 10, temp[pred].max() + 0.1],
                    labels=["Low", "Medium", "High", "Very high"],
                    include_lowest=True,
                )

            avg = temp.groupby("Bucket")[["mood_level", "stress_level", "anxiety_level"]].mean().reset_index()
            avg = avg.dropna(subset=["Bucket"])

            if len(avg) == 0:
                st.info("Not enough variation in this factor for the current filters.")
                continue

            avg_long = avg.melt(id_vars="Bucket", var_name="Outcome", value_name="Value")
            avg_long["Outcome"] = avg_long["Outcome"].str.replace("_", " ").str.title()

            fig = px.line(
                avg_long,
                x="Bucket",
                y="Value",
                color="Outcome",
                markers=True,
                template="simple_white",
                labels={"Bucket": label, "Value": "Average score"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 5: Key Learnings
# -----------------------------------------------------------------------------
with tab5:
    st.header("So… How Bad Is Social Media for Mental Health?")

    st.markdown(
        """
From this **synthetic** dataset, the answer is nuanced:

1. **Heavy use alone is only moderately harmful.**  
   More daily screen time is associated with higher stress and anxiety, but not catastrophically so.
   The effect is noticeable, not overwhelming.

2. **Toxic interactions are the real problem.**  
   Frequent negative interactions are strongly linked to higher stress and anxiety and lower mood.
   In other words, **a hostile digital environment is worse than simply spending a lot of time online.**

3. **Positive interactions help – but cannot fully cancel negativity.**  
   Supportive comments and friendly exchanges lift mood slightly, yet they are not enough to fully
   offset a stream of negative experiences.

4. **Healthy offline habits matter a lot.**  
   Participants who sleep ≥7 hours and move at least ~30 minutes daily report the best mood –
   even when they also spend significant time on social media.

**Conclusion:**  
Social media is **not automatically bad**, but it can become harmful when
- usage is excessive,  
- online environments are toxic, and  
- sleep and physical activity are neglected.

A healthier approach is to:
- limit extreme screen time,
- actively curate digital spaces to avoid negativity,
- and protect sleep and movement as non-negotiable habits.
"""
    )

    st.info(
        "This dashboard is based on a research-style synthetic dataset. Patterns are plausible and "
        "consistent with published work, but they should not be interpreted as clinical or causal evidence."
    )