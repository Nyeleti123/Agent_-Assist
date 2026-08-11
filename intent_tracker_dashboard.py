# STREAMLIT DASHBOARD FOR INTENT TRACKER
# Minimalist Modern Theme with Timeline in File Upload

import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import os

# Set page config
st.set_page_config(page_title="Intent Tracker Dashboard", layout="wide")

# Modern minimalist color palette
colors = {
    'primary': '#2C3E50',
    'secondary': '#5D6D7E',
    'accent': '#3498DB',
    'success': '#27AE60',
    'warning': '#F39C12',
    'info': '#85C1E9',
    'background': '#FFFFFF',
    'card_bg': '#F8F9FA',
    'text': '#2C3E50',
    'text_light': '#7F8C8D',
    'border': '#E5E7EB'
}

# Custom CSS
st.markdown(f"""
<style>
    .stApp {{
        background-color: {colors['background']};
    }}
    .main-header {{
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid {colors['border']};
    }}
    .metric-card {{
        background-color: {colors['card_bg']};
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid {colors['border']};
    }}
    .metric-card h3 {{
        color: {colors['text_light']};
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .metric-card p {{
        color: {colors['primary']};
        font-size: 2rem;
        font-weight: 600;
        margin: 0;
    }}
    .stButton > button {{
        background-color: {colors['accent']};
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    .stButton > button:hover {{
        background-color: {colors['primary']};
        color: white;
    }}
    .stAlert {{
        border-radius: 6px;
        border-left: 3px solid {colors['accent']};
    }}
    .info-box {{
        background-color: {colors['card_bg']};
        padding: 1rem;
        border-radius: 6px;
        border: 1px solid {colors['border']};
        color: {colors['text']};
    }}
    hr {{
        border-color: {colors['border']};
    }}
</style>
""", unsafe_allow_html=True)

# Clean text function
def clean_text_for_model(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load model and vectorizer
@st.cache_resource
def load_model():
    output_folder = r"C:\work_project_1\Output"
    model = joblib.load(os.path.join(output_folder, 'intent_model.pkl'))
    vectorizer = joblib.load(os.path.join(output_folder, 'vectorizer.pkl'))
    return model, vectorizer

class IntentTracker:
    def __init__(self, model, vectorizer, confidence_threshold=0.3):
        self.model = model
        self.vectorizer = vectorizer
        self.confidence_threshold = confidence_threshold
        self.reset()
    
    def reset(self):
        self.intent_history = []
        self.current_intent = None
        self.segment_count = 0
    
    def process_segment(self, segment_text):
        if not segment_text or len(segment_text.strip()) < 5:
            return None, 0, False
        
        cleaned = clean_text_for_model(segment_text)
        if len(cleaned.split()) < 2:
            return None, 0, False
        
        vec = self.vectorizer.transform([cleaned])
        pred = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)
        confidence = max(proba[0])
        
        if confidence < self.confidence_threshold:
            return None, confidence, False
        
        changed = False
        if self.current_intent != pred:
            changed = True
            self.current_intent = pred
        
        self.intent_history.append({
            'segment': self.segment_count,
            'text': segment_text[:100],
            'intent': pred,
            'confidence': confidence,
            'changed': changed
        })
        
        self.segment_count += 1
        return pred, confidence, changed
    
    def get_intent_flow(self):
        flow = []
        for item in self.intent_history:
            if item['changed'] or len(flow) == 0:
                flow.append({
                    'segment': item['segment'],
                    'intent': item['intent'],
                    'confidence': item['confidence']
                })
        return flow
    
    def get_intent_timeline(self):
        timeline = []
        for item in self.intent_history:
            timeline.append({
                'segment': item['segment'],
                'intent': item['intent'],
                'confidence': item['confidence'],
                'changed': item['changed']
            })
        return timeline
    
    def get_summary(self):
        flow = self.get_intent_flow()
        if not flow:
            return "No intents detected"
        summary = f"{flow[0]['intent']}"
        for i in range(1, len(flow)):
            summary += f" → {flow[i]['intent']}"
        return summary

# Load model
model, vectorizer = load_model()
tracker = IntentTracker(model, vectorizer)

# Dashboard header
st.markdown("""
<div class="main-header">
    <h1 style="color: #2C3E50; margin: 0; font-weight: 600;">Intent Tracker</h1>
    <p style="color: #7F8C8D; margin: 0.5rem 0 0 0;">Real-time intent detection for customer conversations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    confidence_threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.3, 0.05)
    tracker.confidence_threshold = confidence_threshold
    
    st.markdown("---")
    st.markdown("### Model info")
    st.markdown(f"**Intent classes:** {len(model.classes_)}")
    st.markdown(f"**Features:** {vectorizer.get_feature_names_out().shape[0]}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Upload conversation files or type text to detect customer intents and track intent changes.")

# Main area - tabs
tab1, tab2, tab3 = st.tabs(["Text Input", "File Upload", "Batch Test"])

# Tab 1: Text Input
with tab1:
    st.markdown("### Real-time tracking")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Reset conversation"):
            tracker.reset()
            st.success("Conversation reset")
    
    with col1:
        user_input = st.text_area("Enter conversation segment:", height=100, key="text_input")
        
        if st.button("Process segment", key="process_btn"):
            if user_input.strip():
                intent, confidence, changed = tracker.process_segment(user_input)
                
                if intent:
                    if changed:
                        st.info(f"Intent changed to: {intent}")
                    else:
                        st.success(f"Intent detected: {intent}")
                    st.metric("Confidence", f"{confidence:.2%}")
                else:
                    st.warning(f"Low confidence ({confidence:.2%}) - intent unclear")
            else:
                st.warning("Please enter text")
    
    st.markdown("### Conversation flow")
    flow = tracker.get_intent_flow()
    
    if flow:
        flow_df = pd.DataFrame(flow)
        fig = go.Figure(data=go.Scatter(
            x=flow_df['segment'],
            y=flow_df['intent'],
            mode='markers+lines',
            text=flow_df['confidence'].apply(lambda x: f"Conf: {x:.2%}"),
            marker=dict(size=10, color=flow_df['confidence'], colorscale='Blues', 
                        line=dict(color='white', width=1)),
            line=dict(color='#3498DB', width=2)
        ))
        fig.update_layout(
            title="Intent flow over conversation",
            xaxis_title="Segment",
            yaxis_title="Intent",
            plot_bgcolor='white',
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"<div class='info-box'>{tracker.get_summary()}</div>", unsafe_allow_html=True)
    else:
        st.info("No intents detected yet")

# Tab 2: File Upload with Timeline
with tab2:
    st.markdown("### Upload conversation file")
    
    file_format = st.radio("File format", ["CSV", "JSON"], horizontal=True)
    
    uploaded_file = st.file_uploader(f"Choose a {file_format} file", type=[file_format.lower()])
    
    if uploaded_file is not None:
        if file_format == "CSV":
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_json(uploaded_file)
        
        st.markdown("#### Data preview")
        st.dataframe(df.head())
        
        text_column = st.selectbox("Select conversation text column", df.columns)
        
        if st.button("Process file", key="file_process"):
            tracker.reset()
            results = []
            all_timeline_data = []
            
            for idx, row in df.iterrows():
                text = str(row[text_column])
                segments = text.split('. ')
                
                for seg_idx, seg in enumerate(segments[:30]):
                    if len(seg.strip()) > 10:
                        intent, confidence, changed = tracker.process_segment(seg)
                        if intent:
                            results.append({
                                'conversation_id': idx,
                                'segment': tracker.segment_count,
                                'intent': intent,
                                'confidence': confidence
                            })
                            all_timeline_data.append({
                                'conversation_id': idx,
                                'segment_index': seg_idx,
                                'segment_text': seg[:100],
                                'intent': intent,
                                'confidence': confidence,
                                'changed': changed
                            })
            
            if results:
                results_df = pd.DataFrame(results)
                timeline_df = pd.DataFrame(all_timeline_data)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Segments processed</h3>
                        <p>{len(results)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Unique intents</h3>
                        <p>{results_df['intent'].nunique()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Avg confidence</h3>
                        <p>{results_df['confidence'].mean():.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Intent distribution chart
                intent_counts = results_df['intent'].value_counts()
                fig1 = px.bar(
                    x=intent_counts.index, 
                    y=intent_counts.values, 
                    title="Intent distribution",
                    color=intent_counts.values,
                    color_continuous_scale='Blues'
                )
                fig1.update_layout(
                    plot_bgcolor='white',
                    xaxis_title="Intent",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig1, use_container_width=True)
                
                # Timeline visualization - Confidence over time
                st.markdown("#### Intent timeline analysis")
                
                fig2 = px.line(
                    timeline_df,
                    x='segment_index',
                    y='confidence',
                    color='intent',
                    title="Confidence over conversation segments",
                    markers=True
                )
                fig2.update_layout(
                    plot_bgcolor='white',
                    xaxis_title="Segment number",
                    yaxis_title="Confidence score",
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Timeline bar chart
                fig3 = px.bar(
                    timeline_df,
                    x='segment_index',
                    y='confidence',
                    color='intent',
                    title="Intent confidence by segment",
                    text='intent'
                )
                fig3.update_traces(textposition='outside')
                fig3.update_layout(
                    plot_bgcolor='white',
                    xaxis_title="Segment number",
                    yaxis_title="Confidence score",
                    height=450
                )
                st.plotly_chart(fig3, use_container_width=True)
                
                # Intent flow sequence
                unique_intents_in_order = []
                for _, row in timeline_df.iterrows():
                    if not unique_intents_in_order or unique_intents_in_order[-1] != row['intent']:
                        unique_intents_in_order.append(row['intent'])
                
                intent_sequence = ' → '.join(unique_intents_in_order)
                st.markdown(f"<div class='info-box'><strong>Intent flow:</strong> {intent_sequence}</div>", unsafe_allow_html=True)
                
                # Detailed timeline table
                with st.expander("View detailed timeline"):
                    st.dataframe(timeline_df)
                
                # Per conversation breakdown
                st.markdown("#### Per conversation breakdown")
                conv_summary = results_df.groupby('conversation_id').agg({
                    'intent': lambda x: ' → '.join(x.unique()),
                    'confidence': 'mean'
                }).reset_index()
                conv_summary.columns = ['Conversation', 'Intent flow', 'Avg confidence']
                st.dataframe(conv_summary)
                
                # Download buttons
                col1, col2 = st.columns(2)
                with col1:
                    csv_results = results_df.to_csv(index=False)
                    st.download_button("Download detection results", csv_results, "intent_results.csv", "text/csv")
                with col2:
                    csv_timeline = timeline_df.to_csv(index=False)
                    st.download_button("Download timeline data", csv_timeline, "intent_timeline.csv", "text/csv")
            else:
                st.warning("No intents detected in the uploaded file")

# Tab 3: Batch Test
with tab3:
    st.markdown("### Batch test on unseen data")
    
    test_file_path = r"C:\work_project_1\Output\test_data_full.csv"
    
    if os.path.exists(test_file_path):
        test_df = pd.read_csv(test_file_path)
        
        st.markdown(f"**Total conversations:** {len(test_df)}")
        
        num_samples = st.slider("Conversations to test", 1, min(30, len(test_df)), 5)
        
        if st.button("Run batch test", key="batch_test"):
            results_summary = []
            
            for idx, row in test_df.head(num_samples).iterrows():
                tracker.reset()
                conversation = row['text']
                actual_intent = row['intent']
                segments = conversation.split('. ')
                
                final_intent = None
                for seg in segments[:15]:
                    if len(seg.strip()) > 10:
                        intent, confidence, changed = tracker.process_segment(seg)
                        if intent:
                            final_intent = intent
                
                results_summary.append({
                    'conversation': idx,
                    'actual': actual_intent,
                    'predicted': final_intent if final_intent else "None",
                    'correct': final_intent == actual_intent if final_intent else False
                })
            
            summary_df = pd.DataFrame(results_summary)
            
            st.markdown("#### Results")
            st.dataframe(summary_df)
            
            accuracy = summary_df['correct'].sum() / len(summary_df)
            st.metric("Batch accuracy", f"{accuracy:.2%}")
            
            cm_data = pd.crosstab(summary_df['actual'], summary_df['predicted'])
            if not cm_data.empty:
                st.markdown("#### Confusion matrix")
                st.dataframe(cm_data)
    else:
        st.error(f"Test file not found")