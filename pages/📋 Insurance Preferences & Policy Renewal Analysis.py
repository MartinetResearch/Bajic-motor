import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from google.genai import types


st.set_page_config(layout="wide")
st.title("📋 Insurance Preferences & Policy Renewal Analysis")

# Load data
@st.cache_data
def load_data():
    return pd.read_excel("combined_insurance_data.xlsx")

df = load_data()

# Weighted chart function
def plot_weighted_chart(rank_cols, weights, title):
    scores = {}
    for col, weight in zip(rank_cols, weights):
        counts = df[col].value_counts().fillna(0) * weight
        for option, score in counts.items():
            scores[option] = scores.get(option, 0) + score
    score_df = pd.DataFrame(list(scores.items()), columns=["Option", "Weighted Score"])
    score_df = score_df.sort_values(by="Weighted Score", ascending=False)
    fig = px.bar(score_df, x="Option", y="Weighted Score", title=title,
                 labels={"Option": "Policy Renewal Reason"}, height=500)
    fig.update_layout(xaxis_tickangle=-45)
    return fig, score_df

# Gemini-based LLM function
def generate_llm_insight(prompt):
    client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(system_instruction="You are a data analyst summarizing insurance user survey data."),
        contents=[prompt],
    )
    return response.text.strip()

# Filters
city_col, age_col, edu_col, job_col = "City", "Age Group", "Education", "Occupation"
with st.sidebar:
    st.header("📋 Filters")
    selected_city = st.selectbox("City", ["All"] + sorted(df[city_col].dropna().unique()))
    selected_age = st.selectbox("Age Group", ["All"] + sorted(df[age_col].dropna().unique()))
    selected_edu = st.selectbox("Education", ["All"] + sorted(df[edu_col].dropna().unique()))
    selected_job = st.selectbox("Occupation", ["All"] + sorted(df[job_col].dropna().unique()))

for col, selected in zip([city_col, age_col, edu_col, job_col], [selected_city, selected_age, selected_edu, selected_job]):
    if selected != "All":
        df = df[df[col] == selected]

selected_section = st.radio("📊 Select Section", ["🎯 Policy Renewal Reasons", "📱 Digital Preferences", "📶 Digital Readiness"], horizontal=True)

if selected_section == "📱 Digital Preferences":
    col1, col2 = st.columns(2)
    with col1:
        # st.subheader("📱 Digital Channel Preference")

        #1
        # Count how many times each option was ranked 1st, 2nd, 3rd
        q12b_rank1 = df['[Rank 1] - Q12b'].value_counts()
        q12b_rank2 = df['[Rank 2] - Q12b'].value_counts()
        q12b_rank3 = df['[Rank 3] - Q12b'].value_counts()

        # Combine into a DataFrame
        q12b_stack_df = pd.DataFrame({
            'Rank 1': q12b_rank1,
            'Rank 2': q12b_rank2,
            'Rank 3': q12b_rank3
        }).fillna(0)

        # Reset index and melt for plotly
        q12b_stack_df = q12b_stack_df.reset_index().rename(columns={'index': 'Option'})
        q12b_melted = q12b_stack_df.melt(id_vars='Option', var_name='Rank', value_name='Count')

        label_map = {
            'User Friendly': 'User Friendly',
            'One stop solution': 'One Stop Solution',
            'Easy to connect with Help Center': 'Easy Support Access',
            'Seamless buying journey': 'Easy Purchase Journey',
            'Value added benefits to user': 'Value-Added Benefits',
            'Locate Us (Branch/Network Garages/Network Hospitals)': 'Easy Location Access'
        }

        # Apply mapping to clean Q12b options
        q12b_stack_df["Option"] = q12b_stack_df["Option"].map(label_map)
        q12b_melted["Option"] = q12b_melted["Option"].map(label_map)

        # Plot stacked bar
        fig_q12b_stacked = px.bar(q12b_melted, x="Option", y="Count", color="Rank",
                                title="Ranking of Digital Preferences by Users", height=500)
        fig_q12b_stacked.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_q12b_stacked, use_container_width=True)
        
        #2
        # Select columns with digital channels
        digital_cols = [col for col in df.columns if any(x in col for x in ["WhatsApp", "App", "Website", "Chatbot", "Call Centre"])]

        # Summarize values
        digital_summary = df[digital_cols].apply(pd.to_numeric, errors='coerce').sum().reset_index()
        digital_summary.columns = ["Channel", "Count"]

        # Sort the data for better visualization
        digital_summary = digital_summary.sort_values(by="Count", ascending=False)

        # Split by label length
        short_labels = digital_summary[digital_summary["Channel"].str.len() < 30]

        # Plot 1: Short labels
        fig1 = px.bar(short_labels, x="Channel", y="Count", title="Digital Channel Preference ", color="Channel")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        #3
        # Q12b Chart
        # Count weighted scores manually
        q12b_weights = {'[Rank 1] - Q12b': 3, '[Rank 2] - Q12b': 2, '[Rank 3] - Q12b': 1}
        q12b_score_dict = {}

        for col, weight in q12b_weights.items():
            scores = df[col].value_counts().fillna(0) * weight
            for option, score in scores.items():
                q12b_score_dict[option] = q12b_score_dict.get(option, 0) + score

        # Convert to DataFrame
        q12b_scores_df = pd.DataFrame(list(q12b_score_dict.items()), columns=["Option", "Weighted Score"])

        # Label Cleanup
        label_map = {
            'User Friendly': 'User Friendly',
            'One stop solution': 'One Stop Solution',
            'Easy to connect with Help Center': 'Easy Support Access',
            'Seamless buying journey': 'Easy Purchase Journey',
            'Value added benefits to user': 'Value-Added Benefits',
            'Locate Us (Branch/Network Garages/Network Hospitals)': 'Easy Location Access'
        }
        q12b_scores_df["Option"] = q12b_scores_df["Option"].map(label_map).dropna()

        # Sort by score
        q12b_scores_df = q12b_scores_df.sort_values(by="Weighted Score", ascending=False)

        # Plot
        fig_q12b_weighted = px.bar(q12b_scores_df,
                                x="Option", y="Weighted Score",
                                title="Weighted Scores for Digital Preferences",
                                labels={"Option": "Policy Renewal Reason"},
                                height=500)
        fig_q12b_weighted.update_layout(xaxis_tickangle=-45)

        # Display in Streamlit
        st.plotly_chart(fig_q12b_weighted, use_container_width=True)
        
        st.subheader("🧠 Gemini Insights")
        digital_prompt = f"""
        Based on this following:
        
        digital preference data (total counts):
        {digital_summary.to_string(index=False)}

        Ranking of Digital Preferences by Users
        {q12b_melted.to_string(index=False)}
        
        Weighted Scores for Digital Preferences
        {q12b_scores_df.to_string(index=False)}

        Summarize 2–3 key insights in simple language.
        """
        insight = generate_llm_insight(digital_prompt)
        st.markdown(f"**Insight:**\n{insight}")

elif selected_section == "🎯 Policy Renewal Reasons":
    col1, col2 = st.columns(2)
    with col1:
        # st.subheader("🎯 Policy Renewal Reasons (Weighted)")
        
        #1 Policy renewal reasons Weighted Ranking Score
        q10_cols = ['[Rank 1] - Q10', '[Rank 2] - Q10', '[Rank 3] - Q10', '[Rank 4] - Q10']
        q10_weights = [4, 3, 2, 1]
        fig_q10, q10_an = plot_weighted_chart(q10_cols, q10_weights, "Weighted Scores for Policy Renewal Reasons")
        st.plotly_chart(fig_q10, use_container_width=True)

        #2 Stacked bar chart of rankings

        # Count each rank occurrence
        q10_rank1 = df['[Rank 1] - Q10'].value_counts()
        q10_rank2 = df['[Rank 2] - Q10'].value_counts()
        q10_rank3 = df['[Rank 3] - Q10'].value_counts()
        q10_rank4 = df['[Rank 4] - Q10'].value_counts()

        # Combine into a single DataFrame
        q10_stack_df = pd.DataFrame({
            'Rank 1': q10_rank1,
            'Rank 2': q10_rank2,
            'Rank 3': q10_rank3,
            'Rank 4': q10_rank4
        }).fillna(0)

        # Reset index for Plotly
        q10_stack_df = q10_stack_df.reset_index().rename(columns={'index': 'Option'})

        # Melt the DataFrame for stacked bar
        q10_melted = q10_stack_df.melt(id_vars='Option', var_name='Rank', value_name='Count')

        # Plotly Stacked Bar
        fig_q10_stacked = px.bar(q10_melted, x="Option", y="Count", color="Rank", 
                                title="User Rankings for Policy Renewal Reasons", height=500)
        fig_q10_stacked.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_q10_stacked, use_container_width=True)

    with col2:
        st.subheader("🧠 Gemini Insights")
        renew_prompt = f"""
        Based on the following: 

        Policy renewal reasons:
        {q10_melted.to_string(index=False)}

        policy renewal reasons Weighted Ranking Score ranking scores:
        {q10_an.to_string(index=False)}

        Summarize the key reasons users renew their policies.
        """
        insight = generate_llm_insight(renew_prompt)
        st.markdown(f"**Insight:**\n{insight}")

elif selected_section == "📶 Digital Readiness":
    col1, col2 = st.columns(2)
    with col1:
        # st.subheader("📶 Digital Readiness Score by Age")
        digital_cols = [col for col in df.columns if any(x in col for x in ["WhatsApp", "App", "Website", "Chatbot", "Call Centre"])]
        df["digital_score"] = df[digital_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        readiness_scores = df.groupby(age_col)["digital_score"].mean().sort_values().reset_index()
        fig = px.bar(readiness_scores, x=age_col, y="digital_score", title="Average Digital Readiness Score by Age", color="digital_score", color_continuous_scale="Purples")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("🧠 Gemini Insights")
        readiness_prompt = f"""
        Based on the average digital readiness score by age group:
        {readiness_scores.to_string(index=False)}
        Provide 2-3 insights about digital channel adoption.
        """
        insight = generate_llm_insight(readiness_prompt)
        st.markdown(f"**Insight:**\n{insight}")
