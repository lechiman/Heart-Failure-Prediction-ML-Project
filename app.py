"""
Heart Failure Prediction Dashboard
A comprehensive Streamlit application for visualizing heart failure prediction model performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Failure Prediction Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #e74c3c;
        margin: 1rem 0;
    }
    .key-finding {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .key-finding:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .key-finding h3 {
        color: #1a202c;
        margin-top: 0;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .key-finding p {
        color: #2d3748;
        line-height: 1.6;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
    }
    .risk-low {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .feature-contribution {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Home", "📊 Data Overview", "🎯 Model Performance", "🔮 Risk Prediction Tool"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This dashboard presents a comprehensive analysis of heart failure 
    prediction using machine learning models.
    
    **Dataset:** Heart Failure Clinical Records  
    **Patients:** 299  
    **Features:** 12 clinical features  
    **Target:** Death Event (Mortality)
    """
)

# Load data (cached for performance)
@st.cache_data
def load_data():
    """Load the heart failure dataset"""
    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    return df

@st.cache_data
def load_model_comparison():
    """Load model comparison metrics"""
    return pd.read_csv('models/model_comparison.csv')

@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {
        'Logistic Regression': joblib.load('models/logistic_regression_model.pkl'),
        'Random Forest': joblib.load('models/random_forest_model.pkl'),
        'XGBoost': joblib.load('models/xgboost_model.pkl'),
        'LightGBM': joblib.load('models/lightgbm_model.pkl')
    }
    scaler = joblib.load('models/scaler.pkl')
    return models, scaler


# =====================================================================
# PAGE 1: HOME
# =====================================================================
if page == "🏠 Home":
    # Main header
    st.markdown('<h1 class="main-header">❤️ Heart Failure Prediction System</h1>', unsafe_allow_html=True)
    
    # Project description
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; margin: 2rem 0;">
        <h2 style="color: white; margin-bottom: 1rem;">🎯 Project Overview</h2>
        <p style="font-size: 1.1rem; line-height: 1.8;">
        This project develops and evaluates machine learning models to predict mortality risk 
        in heart failure patients. Using clinical records from 299 patients, we trained and 
        compared four state-of-the-art models to identify patients at high risk of death, 
        enabling early intervention and improved patient care.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key findings section
    st.markdown('<h2 class="sub-header">Key Findings</h2>', unsafe_allow_html=True)
    
    # Load data for statistics
    df = load_data()
    comparison_df = load_model_comparison()
    
    # Best model identification
    best_model_idx = comparison_df['Test ROC-AUC'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_model_auc = comparison_df.loc[best_model_idx, 'Test ROC-AUC']
    
    # Key findings in cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="key-finding">
            <h3>Best Performing Model</h3>
            <p style="font-size: 1.2rem; font-weight: bold; color: #4c51bf;">
                {model_name}
            </p>
            <p>Achieved an outstanding <strong>ROC-AUC score of {auc:.4f}</strong>, 
            demonstrating excellent discrimination between survival and mortality outcomes.</p>
        </div>
        """.format(model_name=best_model_name, auc=best_model_auc), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="key-finding">
            <h3>Critical Predictive Features</h3>
            <p>Analysis revealed that <strong>time (follow-up period)</strong>, 
            <strong>serum creatinine</strong>, and <strong>ejection fraction</strong> 
            are the most important predictors of mortality, accounting for over 60% 
            of the model's predictive power.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="key-finding">
            <h3>Class Imbalance Strategy</h3>
            <p>The dataset shows a {ratio:.1f}:1 ratio of survivors to deaths. 
            We successfully addressed this imbalance using stratified sampling and 
            appropriate evaluation metrics (ROC-AUC, Precision-Recall) to ensure 
            reliable predictions for both classes.</p>
        </div>
        """.format(ratio=df['DEATH_EVENT'].value_counts()[0] / df['DEATH_EVENT'].value_counts()[1]), 
        unsafe_allow_html=True)
        
        st.markdown("""
        <div class="key-finding">
            <h3>Clinical Impact</h3>
            <p>The model correctly identifies <strong>68-79% of at-risk patients</strong> 
            (recall range across models) while maintaining <strong>79-86% precision</strong>, 
            minimizing false alarms and enabling targeted interventions for high-risk patients.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick statistics
    st.markdown('<h2 class="sub-header">Dataset Statistics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Patients",
            value=len(df),
            delta=None
        )
    
    with col2:
        deaths = df['DEATH_EVENT'].sum()
        st.metric(
            label="Deaths",
            value=deaths,
            delta=f"{deaths/len(df)*100:.1f}%"
        )
    
    with col3:
        survivors = len(df) - deaths
        st.metric(
            label="Survivors",
            value=survivors,
            delta=f"{survivors/len(df)*100:.1f}%"
        )
    
    with col4:
        st.metric(
            label="Clinical Features",
            value=len(df.columns) - 1,
            delta=None
        )
    
    # Model comparison overview
    st.markdown('<h2 class="sub-header">Model Performance Overview</h2>', unsafe_allow_html=True)
    
    # Display simplified metrics table
    metrics_display = comparison_df[['Model', 'Test Accuracy', 'Test Precision', 'Test Recall', 'Test ROC-AUC']].copy()
    metrics_display.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'ROC-AUC']
    metrics_display = metrics_display.sort_values('ROC-AUC', ascending=False)
    
    # Format as percentages
    for col in ['Accuracy', 'Precision', 'Recall', 'ROC-AUC']:
        metrics_display[col] = metrics_display[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(
        metrics_display,
        use_container_width=True,
        hide_index=True
    )
    
    # Call to action
    st.markdown("""
    <div style="background-color: #fff3cd; padding: 1.5rem; border-radius: 10px; 
                margin-top: 2rem; border-left: 5px solid #ffc107;">
        <h3 style="color: #856404; margin-top: 0;">🚀 Explore More</h3>
        <p style="color: #856404; margin-bottom: 0;">
            Navigate to <strong>Data Overview</strong> for detailed exploratory analysis 
            or <strong>Model Performance</strong> for in-depth model evaluation and comparison.
        </p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================
# PAGE 2: DATA OVERVIEW
# =====================================================================
elif page == "📊 Data Overview":
    st.markdown('<h1 class="main-header">📊 Data Overview & Exploratory Analysis</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Dataset statistics section
    st.markdown('<h2 class="sub-header">Dataset Statistics</h2>', unsafe_allow_html=True)
    
    # High-level stats
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Samples", len(df))
    
    with col2:
        st.metric("Features", len(df.columns) - 1)
    
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    with col4:
        st.metric("Duplicates", df.duplicated().sum())
    
    with col5:
        death_rate = df['DEATH_EVENT'].sum() / len(df) * 100
        st.metric("Mortality Rate", f"{death_rate:.1f}%")
    
    # Detailed statistics table
    st.markdown("### Descriptive Statistics")
    
    # Create comprehensive stats
    stats_df = df.describe().T
    stats_df['missing'] = df.isnull().sum()
    stats_df['unique'] = df.nunique()
    
    # Reorder columns
    stats_df = stats_df[['count', 'missing', 'unique', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    # Visualizations section
    st.markdown('<h2 class="sub-header">Key Visualizations</h2>', unsafe_allow_html=True)
    
    # Visualization 1: Target Distribution
    st.markdown("### Target Variable Distribution (Death Event)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        death_counts = df['DEATH_EVENT'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        axes[0].bar(death_counts.index, death_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[0].set_xlabel('Death Event', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribution of Death Events', fontsize=14, fontweight='bold')
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Survived', 'Died'])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add counts on bars
        for idx, val in enumerate(death_counts.values):
            axes[0].text(idx, val + 5, str(val), ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Pie chart
        axes[1].pie(death_counts.values, labels=['Survived', 'Died'], autopct='%1.1f%%',
                   colors=colors, startangle=90, explode=[0.05, 0.05],
                   textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[1].set_title('Death Event Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Class Distribution</h4>
            <p><strong>Survived:</strong> {survived} ({survived_pct:.1f}%)</p>
            <p><strong>Died:</strong> {died} ({died_pct:.1f}%)</p>
            <p><strong>Imbalance Ratio:</strong> {ratio:.2f}:1</p>
            <hr>
            <p style="font-size: 0.9rem; color: #555;">
            The dataset shows moderate class imbalance, which we addressed 
            using stratified sampling and appropriate evaluation metrics.
            </p>
        </div>
        """.format(
            survived=death_counts[0],
            survived_pct=death_counts[0]/len(df)*100,
            died=death_counts[1],
            died_pct=death_counts[1]/len(df)*100,
            ratio=death_counts[0]/death_counts[1]
        ), unsafe_allow_html=True)
    
    # Visualization 2: Age Distribution by Outcome
    st.markdown("### Age Distribution by Death Event")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    survived = df[df['DEATH_EVENT'] == 0]['age']
    died = df[df['DEATH_EVENT'] == 1]['age']
    
    # Histogram
    axes[0].hist(survived, bins=20, alpha=0.7, label='Survived', color='#2ecc71', edgecolor='black')
    axes[0].hist(died, bins=20, alpha=0.7, label='Died', color='#e74c3c', edgecolor='black')
    axes[0].set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Age Distribution by Death Event', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot
    df.boxplot(column='age', by='DEATH_EVENT', ax=axes[1], patch_artist=True,
               boxprops=dict(facecolor='lightblue', color='black'),
               medianprops=dict(color='red', linewidth=2),
               whiskerprops=dict(color='black'),
               capprops=dict(color='black'))
    axes[1].set_xlabel('Death Event (0=Survived, 1=Died)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Age (years)', fontsize=12, fontweight='bold')
    axes[1].set_title('Age Distribution Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    plt.suptitle('')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Survived - Mean Age", f"{survived.mean():.1f} years")
    with col2:
        st.metric("Died - Mean Age", f"{died.mean():.1f} years")
    with col3:
        st.metric("Age Difference", f"{died.mean() - survived.mean():.1f} years", 
                 delta=f"+{died.mean() - survived.mean():.1f}")
    
    # Visualization 3: Correlation Heatmap
    st.markdown("### Feature Correlation Heatmap")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    correlation_matrix = df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                mask=mask, vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation Heatmap of All Features', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Show top correlations with death event
    st.markdown("### Features Correlated with Death Event")
    
    death_corr = correlation_matrix['DEATH_EVENT'].sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Positive Correlations (Risk Factors):**")
        positive_corr = death_corr[death_corr > 0].sort_values(ascending=False)[1:6]
        for feat, corr in positive_corr.items():
            st.write(f"• **{feat}**: {corr:.3f}")
    
    with col2:
        st.markdown("**Top Negative Correlations (Protective Factors):**")
        negative_corr = death_corr[death_corr < 0].sort_values()[0:5]
        for feat, corr in negative_corr.items():
            st.write(f"• **{feat}**: {corr:.3f}")
    
    # Visualization 4: Ejection Fraction Analysis
    st.markdown("### Ejection Fraction vs Mortality")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    survived_ef = df[df['DEATH_EVENT'] == 0]['ejection_fraction']
    died_ef = df[df['DEATH_EVENT'] == 1]['ejection_fraction']
    
    # Distribution
    axes[0].hist(survived_ef, bins=15, alpha=0.7, label='Survived', color='#2ecc71', edgecolor='black')
    axes[0].hist(died_ef, bins=15, alpha=0.7, label='Died', color='#e74c3c', edgecolor='black')
    axes[0].set_xlabel('Ejection Fraction (%)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Ejection Fraction Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot
    df.boxplot(column='ejection_fraction', by='DEATH_EVENT', ax=axes[1], patch_artist=True,
               boxprops=dict(facecolor='lightcoral', color='black'),
               medianprops=dict(color='darkred', linewidth=2))
    axes[1].set_xlabel('Death Event (0=Survived, 1=Died)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Ejection Fraction (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Ejection Fraction by Outcome', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    plt.suptitle('')
    
    # Violin plot
    sns.violinplot(data=df, x='DEATH_EVENT', y='ejection_fraction', ax=axes[2],
                   palette=['#2ecc71', '#e74c3c'])
    axes[2].set_xlabel('Death Event (0=Survived, 1=Died)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Ejection Fraction (%)', fontsize=12, fontweight='bold')
    axes[2].set_title('Ejection Fraction Distribution (Violin)', fontsize=14, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Survived - Mean EF", f"{survived_ef.mean():.1f}%")
    with col2:
        st.metric("Died - Mean EF", f"{died_ef.mean():.1f}%")
    with col3:
        st.metric("Difference", f"{survived_ef.mean() - died_ef.mean():.1f}%",
                 delta=f"+{survived_ef.mean() - died_ef.mean():.1f}")
    
    # Key insights
    st.markdown("""
    <div style="background-color: #d1ecf1; padding: 1.5rem; border-radius: 10px; 
                margin-top: 2rem; border-left: 5px solid #17a2b8;">
        <h3 style="color: #0c5460; margin-top: 0;">💡 Key Data Insights</h3>
        <ul style="color: #0c5460;">
            <li>Patients who died were on average <strong>{age_diff:.1f} years older</strong> than survivors</li>
            <li>Lower ejection fraction is strongly associated with higher mortality risk</li>
            <li>Time (follow-up period), serum creatinine, and ejection fraction show the strongest 
                correlations with death events</li>
            <li>The dataset is clean with no missing values or duplicates, ready for modeling</li>
        </ul>
    </div>
    """.format(age_diff=died.mean() - survived.mean()), unsafe_allow_html=True)

# =====================================================================
# PAGE 3: MODEL PERFORMANCE
# =====================================================================
elif page == "🎯 Model Performance":
    st.markdown('<h1 class="main-header">🎯 Model Performance Evaluation</h1>', unsafe_allow_html=True)
    
    # Load data and models
    df = load_data()
    comparison_df = load_model_comparison()
    models, scaler = load_models()
    
    # Prepare test data
    from sklearn.model_selection import train_test_split
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)
    
    # Best model identification
    best_model_idx = comparison_df['Test ROC-AUC'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_model = models[best_model_name]
    
    # Display best model info
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
        <h2 style="color: white; margin: 0;">🏆 Best Model: {model_name}</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Selected based on highest Test ROC-AUC Score
        </p>
    </div>
    """.format(model_name=best_model_name), unsafe_allow_html=True)
    
    # =====================================================================
    # ROC CURVES COMPARISON
    # =====================================================================
    st.markdown('<h2 class="sub-header">ROC Curves Comparison</h2>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors
    colors = {
        'Logistic Regression': '#3498db',
        'Random Forest': '#2ecc71',
        'XGBoost': '#e74c3c',
        'LightGBM': '#f39c12'
    }
    
    # Plot ROC curve for each model
    auc_scores = {}
    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        auc_scores[model_name] = roc_auc
        
        # Highlight best model
        lw = 4 if model_name == best_model_name else 3
        alpha = 1.0 if model_name == best_model_name else 0.7
        
        ax.plot(fpr, tpr, color=colors[model_name], lw=lw,
               label=f'{model_name} (AUC = {roc_auc:.4f})', alpha=alpha)
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--',
           label='Random Classifier (AUC = 0.5000)', alpha=0.7)
    
    # Formatting
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves - All Models Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    ax.grid(alpha=0.3, linestyle=':')
    
    # Add shaded regions
    ax.fill_between([0, 1], [0, 1], [1, 1], alpha=0.1, color='green')
    ax.fill_between([0, 1], [0, 0], [0, 1], alpha=0.1, color='red')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # AUC Ranking
    st.markdown("### ROC-AUC Rankings")
    sorted_auc = sorted(auc_scores.items(), key=lambda x: x[1], reverse=True)
    
    col1, col2, col3, col4 = st.columns(4)
    for i, (col, (model_name, score)) in enumerate(zip([col1, col2, col3, col4], sorted_auc)):
        with col:
            medal = ["🥇", "🥈", "🥉", "🏅"][i]
            st.metric(f"{medal} {model_name}", f"{score:.4f}")
    
    # =====================================================================
    # METRICS COMPARISON TABLE
    # =====================================================================
    st.markdown('<h2 class="sub-header">Performance Metrics Comparison</h2>', unsafe_allow_html=True)
    
    # Prepare metrics table
    metrics_display = comparison_df[['Model', 'Test Accuracy', 'Test Precision', 
                                     'Test Recall', 'Test F1', 'Test ROC-AUC']].copy()
    metrics_display.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    metrics_display = metrics_display.sort_values('ROC-AUC', ascending=False)
    
    # Highlight best model
    def highlight_best(row):
        if row['Model'] == best_model_name:
            return ['background-color: #d4edda; font-weight: bold'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        metrics_display.style.apply(highlight_best, axis=1).format({
            'Accuracy': '{:.4f}',
            'Precision': '{:.4f}',
            'Recall': '{:.4f}',
            'F1-Score': '{:.4f}',
            'ROC-AUC': '{:.4f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # Best metrics by category
    st.markdown("### Best Performance by Metric")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    cols = [col1, col2, col3, col4, col5]
    
    for col, metric in zip(cols, metrics_cols):
        with col:
            best_idx = metrics_display[metric].astype(float).idxmax()
            best_model_for_metric = metrics_display.loc[best_idx, 'Model']
            best_value = metrics_display.loc[best_idx, metric]
            st.metric(metric, f"{best_value:.4f}", delta=best_model_for_metric)
    
    # Visual comparison
    st.markdown("### Visual Metrics Comparison")
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Model Performance Metrics Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    colors_list = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for idx, metric in enumerate(metrics_cols):
        ax = axes[idx]
        sorted_data = metrics_display.sort_values(metric, ascending=True)
        
        bars = ax.barh(sorted_data['Model'], sorted_data[metric].astype(float),
                      color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (model, value) in enumerate(zip(sorted_data['Model'], sorted_data[metric])):
            ax.text(float(value) + 0.01, i, f'{float(value):.3f}',
                   va='center', fontweight='bold', fontsize=9)
        
        ax.set_xlabel(metric, fontsize=11, fontweight='bold')
        ax.set_xlim([0, 1.0])
        ax.grid(axis='x', alpha=0.3, linestyle=':')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # =====================================================================
    # BEST MODEL CONFUSION MATRIX
    # =====================================================================
    st.markdown('<h2 class="sub-header">Best Model Confusion Matrix</h2>', unsafe_allow_html=True)
    
    # Generate predictions
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Standard confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[0],
                xticklabels=['Survived (0)', 'Died (1)'],
                yticklabels=['Survived (0)', 'Died (1)'],
                annot_kws={'size': 16, 'fontweight': 'bold'},
                linewidths=2, linecolor='black')
    axes[0].set_title(f'Confusion Matrix\n{best_model_name}', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    
    # 2. Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', cbar=True, ax=axes[1],
                xticklabels=['Survived (0)', 'Died (1)'],
                yticklabels=['Survived (0)', 'Died (1)'],
                annot_kws={'size': 14, 'fontweight': 'bold'},
                linewidths=2, linecolor='black')
    axes[1].set_title('Normalized by Actual\n(Recall Rate)', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Actual', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
    
    # 3. Metrics breakdown
    axes[2].axis('off')
    
    total = cm.sum()
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    details_text = f"""
CONFUSION MATRIX BREAKDOWN
{'=' * 35}

Raw Counts:
  True Negatives (TN):  {tn:>3d}
  False Positives (FP): {fp:>3d}
  False Negatives (FN): {fn:>3d}
  True Positives (TP):  {tp:>3d}
  {'─' * 35}
  Total Predictions:    {total:>3d}

Performance Metrics:
  Accuracy:    {accuracy:.4f} ({accuracy*100:.1f}%)
  Precision:   {precision:.4f} ({precision*100:.1f}%)
  Recall:      {recall:.4f} ({recall*100:.1f}%)
  Specificity: {specificity:.4f} ({specificity*100:.1f}%)
  F1-Score:    {f1:.4f}

Clinical Interpretation:
  • {tp} deaths correctly predicted
  • {tn} survivals correctly predicted
  • {fn} deaths missed (Type II error)
  • {fp} false alarms (Type I error)
"""
    
    axes[2].text(0.05, 0.5, details_text, fontsize=10, fontfamily='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))
    
    plt.suptitle(f'Comprehensive Confusion Matrix Analysis - {best_model_name}',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Key metrics in columns
    st.markdown("### Detailed Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy:.4f}", f"{accuracy*100:.1f}%")
        st.metric("True Positives", tp)
    
    with col2:
        st.metric("Precision", f"{precision:.4f}", f"{precision*100:.1f}%")
        st.metric("True Negatives", tn)
    
    with col3:
        st.metric("Recall (Sensitivity)", f"{recall:.4f}", f"{recall*100:.1f}%")
        st.metric("False Positives", fp)
    
    with col4:
        st.metric("Specificity", f"{specificity:.4f}", f"{specificity*100:.1f}%")
        st.metric("False Negatives", fn)
    
    # Clinical interpretation
    st.markdown("""
    <div style="background-color: #d1ecf1; padding: 1.5rem; border-radius: 10px; 
                margin-top: 2rem; border-left: 5px solid #17a2b8;">
        <h3 style="color: #0c5460; margin-top: 0;">🏥 Clinical Interpretation</h3>
        <ul style="color: #0c5460;">
            <li>The model correctly identifies <strong>{recall_pct:.1f}% of actual deaths</strong> (Recall/Sensitivity)</li>
            <li>When the model predicts death, it's correct <strong>{precision_pct:.1f}% of the time</strong> (Precision)</li>
            <li>The model correctly identifies <strong>{specificity_pct:.1f}% of survivors</strong> (Specificity)</li>
            <li>Overall accuracy: <strong>{accuracy_pct:.1f}%</strong></li>
            <li><strong>{fn} high-risk patients</strong> were incorrectly classified as low-risk (False Negatives)</li>
            <li><strong>{fp} low-risk patients</strong> were incorrectly classified as high-risk (False Positives)</li>
        </ul>
    </div>
    """.format(
        recall_pct=recall*100,
        precision_pct=precision*100,
        specificity_pct=specificity*100,
        accuracy_pct=accuracy*100,
        fn=fn,
        fp=fp
    ), unsafe_allow_html=True)
    
    # Model insights
    st.markdown("""
    <div style="background-color: #fff3cd; padding: 1.5rem; border-radius: 10px; 
                margin-top: 1rem; border-left: 5px solid #ffc107;">
        <h3 style="color: #856404; margin-top: 0;">💡 Key Insights</h3>
        <ul style="color: #856404;">
            <li><strong>{best_model}</strong> outperforms all other models with a ROC-AUC of <strong>{auc:.4f}</strong></li>
            <li>The model balances precision and recall effectively, achieving an F1-Score of <strong>{f1:.4f}</strong></li>
            <li>Low false negative rate ({fn}) minimizes missed high-risk patients</li>
            <li>The model can be used for clinical decision support to identify at-risk patients early</li>
        </ul>
    </div>
    """.format(
        best_model=best_model_name,
        auc=auc_scores[best_model_name],
        f1=f1,
        fn=fn
    ), unsafe_allow_html=True)

# =====================================================================
# PAGE 4: RISK PREDICTION TOOL ⭐
# =====================================================================
elif page == "🔮 Risk Prediction Tool":
    st.markdown('<h1 class="main-header">🔮 Heart Failure Risk Prediction Tool</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1.5rem; border-radius: 15px; color: white; margin: 1rem 0;">
        <h2 style="color: white; margin: 0;">⭐ Interactive Risk Assessment</h2>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Enter patient clinical data below to predict heart failure mortality risk using our best-performing AI model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models, scaler = load_models()
    comparison_df = load_model_comparison()
    
    # Get best model
    best_model_idx = comparison_df['Test ROC-AUC'].idxmax()
    best_model_name = comparison_df.loc[best_model_idx, 'Model']
    best_model = models[best_model_name]
    
    st.info(f"🏆 **Using Best Model:** {best_model_name} (ROC-AUC: {comparison_df.loc[best_model_idx, 'Test ROC-AUC']:.4f})")
    
    # Create two columns for input form
    st.markdown('<h2 class="sub-header">Patient Clinical Data</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Demographics & Vital Signs")
        
        age = st.slider(
            "Age (years)",
            min_value=20,
            max_value=100,
            value=60,
            step=1,
            help="Patient's age in years"
        )
        
        sex = st.selectbox(
            "Sex",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            help="Patient's biological sex"
        )
        
        st.markdown("### Cardiac Measurements")
        
        ejection_fraction = st.slider(
            "Ejection Fraction (%)",
            min_value=10,
            max_value=80,
            value=38,
            step=1,
            help="Percentage of blood leaving the heart at each contraction"
        )
        
        creatinine_phosphokinase = st.number_input(
            "Creatinine Phosphokinase (mcg/L)",
            min_value=20,
            max_value=8000,
            value=250,
            step=10,
            help="Level of CPK enzyme in the blood"
        )
        
        platelets = st.number_input(
            "Platelets (kiloplatelets/mL)",
            min_value=25000,
            max_value=850000,
            value=263000,
            step=1000,
            help="Platelet count in blood"
        )
        
        st.markdown("### Lab Results")
        
        serum_creatinine = st.number_input(
            "Serum Creatinine (mg/dL)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f",
            help="Level of creatinine in the blood"
        )
        
        serum_sodium = st.slider(
            "Serum Sodium (mEq/L)",
            min_value=110,
            max_value=150,
            value=137,
            step=1,
            help="Level of sodium in the blood"
        )
    
    with col2:
        st.markdown("### Medical Conditions")
        
        anaemia = st.checkbox(
            "Anaemia",
            value=False,
            help="Decrease of red blood cells or hemoglobin"
        )
        
        diabetes = st.checkbox(
            "Diabetes",
            value=False,
            help="Patient has diabetes"
        )
        
        high_blood_pressure = st.checkbox(
            "High Blood Pressure",
            value=False,
            help="Patient has hypertension"
        )
        
        smoking = st.checkbox(
            "Smoking",
            value=False,
            help="Patient smokes"
        )
        
        st.markdown("### Follow-up Period")
        
        time = st.number_input(
            "Follow-up Period (days)",
            min_value=1,
            max_value=300,
            value=130,
            step=1,
            help="Follow-up period in days"
        )
        
        st.markdown("---")
        
        # Quick preset buttons
        st.markdown("### Quick Presets")
        
        col_preset1, col_preset2 = st.columns(2)
        
        with col_preset1:
            if st.button("💚 Low Risk Example", use_container_width=True):
                st.session_state.preset = "low"
                st.rerun()
        
        with col_preset2:
            if st.button("❤️ High Risk Example", use_container_width=True):
                st.session_state.preset = "high"
                st.rerun()
    
    # Apply presets if selected
    if 'preset' in st.session_state:
        if st.session_state.preset == "low":
            age = 45
            sex = 1
            ejection_fraction = 50
            creatinine_phosphokinase = 120
            platelets = 300000
            serum_creatinine = 0.9
            serum_sodium = 139
            anaemia = False
            diabetes = False
            high_blood_pressure = False
            smoking = False
            time = 200
        elif st.session_state.preset == "high":
            age = 75
            sex = 1
            ejection_fraction = 20
            creatinine_phosphokinase = 580
            platelets = 200000
            serum_creatinine = 2.5
            serum_sodium = 125
            anaemia = True
            diabetes = True
            high_blood_pressure = True
            smoking = True
            time = 30
        del st.session_state.preset
    
    # Predict button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Risk", use_container_width=True, type="primary")
    
    # Prediction logic
    if predict_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'age': [age],
            'anaemia': [int(anaemia)],
            'creatinine_phosphokinase': [creatinine_phosphokinase],
            'diabetes': [int(diabetes)],
            'ejection_fraction': [ejection_fraction],
            'high_blood_pressure': [int(high_blood_pressure)],
            'platelets': [platelets],
            'serum_creatinine': [serum_creatinine],
            'serum_sodium': [serum_sodium],
            'sex': [sex],
            'smoking': [int(smoking)],
            'time': [time]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        risk_probability = best_model.predict_proba(input_scaled)[0][1]
        risk_percentage = risk_probability * 100
        
        # Determine risk category
        if risk_probability < 0.3:
            risk_category = "Low Risk"
            risk_class = "risk-low"
            risk_emoji = "💚"
            risk_color = "#2ecc71"
        elif risk_probability < 0.6:
            risk_category = "Medium Risk"
            risk_class = "risk-medium"
            risk_emoji = "⚠️"
            risk_color = "#f39c12"
        else:
            risk_category = "High Risk"
            risk_class = "risk-high"
            risk_emoji = "❤️"
            risk_color = "#e74c3c"
        
        # Display results
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        # Big risk display
        st.markdown(f"""
        <div class="{risk_class}">
            <div style="font-size: 1.2rem; margin-bottom: 0.5rem;">Mortality Risk Probability</div>
            <div style="font-size: 3.5rem; font-weight: bold;">{risk_emoji} {risk_percentage:.1f}%</div>
            <div style="font-size: 1.5rem; margin-top: 0.5rem;">{risk_category}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk interpretation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Risk Category",
                risk_category,
                delta=None
            )
        
        with col2:
            st.metric(
                "Confidence",
                f"{max(risk_probability, 1-risk_probability)*100:.1f}%",
                delta=None
            )
        
        with col3:
            survival_prob = (1 - risk_probability) * 100
            st.metric(
                "Survival Probability",
                f"{survival_prob:.1f}%",
                delta=None
            )
        # Clinical recommendations
        st.markdown('<h2 class="sub-header">Clinical Recommendations</h2>', unsafe_allow_html=True)
        
        if risk_probability < 0.3:
            recommendations = """
            <div style="background-color: #d4edda; padding: 1.5rem; border-radius: 10px; 
                        border-left: 5px solid #28a745;">
                <h3 style="color: #155724; margin-top: 0;">✅ Low Risk Patient</h3>
                <ul style="color: #155724;">
                    <li>Continue routine monitoring and follow-up care</li>
                    <li>Maintain current medication regimen</li>
                    <li>Encourage healthy lifestyle habits (diet, exercise)</li>
                    <li>Schedule regular check-ups as per standard protocol</li>
                </ul>
            </div>
            """
        elif risk_probability < 0.6:
            recommendations = """
            <div style="background-color: #fff3cd; padding: 1.5rem; border-radius: 10px; 
                        border-left: 5px solid #ffc107;">
                <h3 style="color: #856404; margin-top: 0;">⚠️ Medium Risk Patient</h3>
                <ul style="color: #856404;">
                    <li>Consider more frequent monitoring and follow-up visits</li>
                    <li>Review and optimize medication regimen</li>
                    <li>Address modifiable risk factors (smoking, hypertension, diabetes)</li>
                    <li>Consider additional diagnostic tests if warranted</li>
                    <li>Provide patient education on warning signs</li>
                </ul>
            </div>
            """
        else:
            recommendations = """
            <div style="background-color: #f8d7da; padding: 1.5rem; border-radius: 10px; 
                        border-left: 5px solid #dc3545;">
                <h3 style="color: #721c24; margin-top: 0;">🚨 High Risk Patient</h3>
                <ul style="color: #721c24;">
                    <li><strong>URGENT:</strong> Consider immediate clinical review</li>
                    <li>Intensify monitoring and increase follow-up frequency</li>
                    <li>Review and aggressively optimize treatment plan</li>
                    <li>Address all modifiable risk factors immediately</li>
                    <li>Consider advanced interventions or specialist referral</li>
                    <li>Provide comprehensive patient and family education</li>
                    <li>Ensure close communication with care team</li>
                </ul>
            </div>
            """
        
        st.markdown(recommendations, unsafe_allow_html=True)
        
        # Disclaimer
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; 
                    margin-top: 2rem; border-left: 3px solid #6c757d;">
            <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                <strong>⚠️ Medical Disclaimer:</strong> This prediction tool is for educational and research purposes only. 
                It should not replace professional medical judgment or clinical assessment. 
                All treatment decisions should be made by qualified healthcare professionals 
                based on comprehensive patient evaluation.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 1rem;">
    <p>Heart Failure Prediction Dashboard | Built with Streamlit | © 2024</p>
</div>
""", unsafe_allow_html=True)

