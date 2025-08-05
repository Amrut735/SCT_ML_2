import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for neutral theme styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 0.5rem 0;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #e55a5a, #3db8b0);
        color: white;
    }
    
    .stExpander {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: 2px solid #4ECDC4;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .stExpander .streamlit-expanderHeader {
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 16px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        padding: 15px;
        background: rgba(255,255,255,0.1);
        border-radius: 8px 8px 0 0;
    }
    
    .stExpander .streamlit-expanderContent {
        color: #FF0000 !important;
        background: rgba(255,255,255,0.95);
        padding: 20px;
        border-radius: 0 0 8px 8px;
        margin: 0;
    }
    
    .stDataFrame {
        background: #FFFFFF;
        border: 1px solid #DEE2E6;
    }
    
    /* Make all text in the app more visible */
    .main .block-container {
        color: #2c3e50;
    }
    
    .main .block-container p, .main .block-container div {
        color: #2c3e50 !important;
    }
    
    /* Style for metric cards */
    .metric-container {
        color: #2c3e50 !important;
    }
    
    /* Make cluster analysis text red for better visibility */
    .stExpander .streamlit-expanderContent p,
    .stExpander .streamlit-expanderContent div,
    .stExpander .streamlit-expanderContent span,
    .stExpander .streamlit-expanderContent .markdown-text-container,
    .stExpander .streamlit-expanderContent .element-container,
    .stExpander .streamlit-expanderContent .stMarkdown {
        color: #FF0000 !important;
    }
    
    /* Target specific cluster analysis content */
    .stExpander .streamlit-expanderContent .markdown-text-container {
        color: #FF0000 !important;
    }
    
    /* Force red color for all text in expanders */
    .stExpander .streamlit-expanderContent * {
        color: #FF0000 !important;
    }
    
    /* Override any other color settings */
    .stExpander .streamlit-expanderContent .stMarkdown p,
    .stExpander .streamlit-expanderContent .stMarkdown div,
    .stExpander .streamlit-expanderContent .stMarkdown span {
        color: #FF0000 !important;
    }
    
    /* Add hover effects for expanders */
    .stExpander:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    
    /* Style for the expander content text */
    .stExpander .streamlit-expanderContent div[style*="color: #FF0000"] {
        background: rgba(255,255,255,0.8);
        padding: 8px 12px;
        border-radius: 6px;
        margin: 5px 0;
        border-left: 4px solid #FF6B6B;
    }
</style>
""", unsafe_allow_html=True)

# Vibrant color palette
NEUTRAL_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
CLUSTER_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

@st.cache_data
def generate_sample_data(n_customers=200, age_range=(20, 70), income_range=(3, 25), 
                        spending_range=(50, 100), random_seed=42):
    """Generate synthetic customer data"""
    np.random.seed(random_seed)
    
    # Generate data with realistic correlations
    age = np.random.normal((age_range[0] + age_range[1])/2, 15, n_customers)
    age = np.clip(age, age_range[0], age_range[1])
    
    # Income correlated with age (older = higher income)
    income_base = np.random.normal((income_range[0] + income_range[1])/2, 5, n_customers)
    income = income_base + (age - 40) * 0.3
    income = np.clip(income, income_range[0], income_range[1])
    
    # Spending score (inverse correlation with age, positive with income)
    spending_base = np.random.normal((spending_range[0] + spending_range[1])/2, 20, n_customers)
    spending = spending_base - (age - 40) * 0.5 + (income - 10) * 2
    spending = np.clip(spending, spending_range[0], spending_range[1])
    
    data = pd.DataFrame({
        'Age': age.round(1),
        'Annual_Income_Lakhs': income.round(1),
        'Spending_Score': spending.round(1)
    })
    
    return data

def find_optimal_k(data, max_k=10):
    """Find optimal number of clusters using Elbow method and Silhouette analysis"""
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
        if k > 1:
            silhouette_avg = silhouette_score(data, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
    
    # Find optimal k using elbow method
    optimal_k = 3  # Default
    for i in range(1, len(inertias) - 1):
        if inertias[i-1] - inertias[i] > inertias[i] - inertias[i+1]:
            optimal_k = k_range[i]
            break
    
    return optimal_k, inertias, silhouette_scores, list(k_range)

def perform_clustering(data, k):
    """Perform K-Means clustering"""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(data)
    
    return {
        'labels': labels,
        'centroids': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_
    }

def create_correlation_heatmap(data):
    """Create correlation heatmap"""
    corr_matrix = data.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="Feature Correlation Matrix"
    )
    
    fig.update_layout(
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        font=dict(color='#2c3e50', size=14),
        title_font=dict(size=18, color='#2c3e50'),
        xaxis=dict(
            title_font=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=12, color='#2c3e50')
        ),
        yaxis=dict(
            title_font=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=12, color='#2c3e50')
        )
    )
    
    return fig

def create_elbow_silhouette_plots(inertias, silhouette_scores, k_range):
    """Create elbow and silhouette plots"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Elbow Method', 'Silhouette Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Elbow plot
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(color='#FF6B6B', width=3),
            marker=dict(size=8, color='#FF6B6B')
        ),
        row=1, col=1
    )
    
    # Silhouette plot
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=8, color='#4ECDC4')
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        font=dict(color='#2c3e50', size=14),
        title_font=dict(size=18, color='#2c3e50')
    )
    
    fig.update_xaxes(
        title_text="Number of Clusters (k)", 
        row=1, col=1,
        title_font=dict(size=14, color='#2c3e50'),
        tickfont=dict(size=12, color='#2c3e50')
    )
    fig.update_xaxes(
        title_text="Number of Clusters (k)", 
        row=1, col=2,
        title_font=dict(size=14, color='#2c3e50'),
        tickfont=dict(size=12, color='#2c3e50')
    )
    fig.update_yaxes(
        title_text="Inertia", 
        row=1, col=1,
        title_font=dict(size=14, color='#2c3e50'),
        tickfont=dict(size=12, color='#2c3e50')
    )
    fig.update_yaxes(
        title_text="Silhouette Score", 
        row=1, col=2,
        title_font=dict(size=14, color='#2c3e50'),
        tickfont=dict(size=12, color='#2c3e50')
    )
    
    return fig

def create_cluster_scatter_plot(data, labels, centroids, feature_names):
    """Create 3D scatter plot of clusters"""
    if len(feature_names) >= 2:
        x_feature, y_feature = feature_names[0], feature_names[1]
        
        fig = px.scatter(
            data,
            x=x_feature,
            y=y_feature,
            color=labels,
            title=f"Cluster Visualization: {x_feature} vs {y_feature}",
            color_discrete_sequence=CLUSTER_COLORS,
            opacity=0.8
        )
        
        # Add centroids
        fig.add_trace(
            go.Scatter(
                x=centroids[:, 0],
                y=centroids[:, 1],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=15,
                    color='#FF6B6B',
                    line=dict(width=3, color='white')
                ),
                name='Cluster Centers',
                showlegend=True
            )
        )
        
        fig.update_layout(
            plot_bgcolor='#f8f9fa',
            paper_bgcolor='#f8f9fa',
            font=dict(color='#2c3e50', size=14),
            title_font=dict(size=18, color='#2c3e50'),
            xaxis=dict(
                title_font=dict(size=14, color='#2c3e50'),
                tickfont=dict(size=12, color='#2c3e50')
            ),
            yaxis=dict(
                title_font=dict(size=14, color='#2c3e50'),
                tickfont=dict(size=12, color='#2c3e50')
            ),
            legend=dict(
                bgcolor='rgba(248,249,250,0.9)',
                bordercolor='#4ECDC4',
                font=dict(size=12, color='#2c3e50')
            )
        )
        
        return fig
    else:
        return None

def create_cluster_distribution_plots(data, labels, feature_names):
    """Create distribution plots for each feature by cluster"""
    fig = make_subplots(
        rows=1, cols=len(feature_names),
        subplot_titles=feature_names,
        specs=[[{"type": "histogram"} for _ in feature_names]]
    )
    
    for i, feature in enumerate(feature_names):
        for cluster_id in np.unique(labels):
            cluster_data = data[labels == cluster_id][feature]
            fig.add_trace(
                go.Histogram(
                    x=cluster_data,
                    name=f'Cluster {cluster_id}',
                    opacity=0.7,
                    marker_color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                    showlegend=(i == 0)
                ),
                row=1, col=i+1
            )
    
    fig.update_layout(
        height=400,
        title_text="Feature Distribution by Cluster",
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='#f8f9fa',
        font=dict(color='#2c3e50', size=14),
        title_font=dict(size=18, color='#2c3e50')
    )
    
    # Update axis labels for all subplots
    for i in range(len(feature_names)):
        fig.update_xaxes(
            title_text=feature_names[i],
            row=1, col=i+1,
            title_font=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=12, color='#2c3e50')
        )
        fig.update_yaxes(
            title_text="Count",
            row=1, col=i+1,
            title_font=dict(size=14, color='#2c3e50'),
            tickfont=dict(size=12, color='#2c3e50')
        )
    
    return fig

def calculate_cluster_statistics(data, labels, feature_names):
    """Calculate comprehensive cluster statistics"""
    cluster_stats = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        mask = labels == label
        cluster_data = data[mask]
        
        stats = {}
        for feature in feature_names:
            feature_data = cluster_data[feature]
            stats[feature] = {
                'mean': float(feature_data.mean()),
                'median': float(feature_data.median()),
                'std': float(feature_data.std()),
                'min': float(feature_data.min()),
                'max': float(feature_data.max()),
                'count': int(len(feature_data))
            }
        
        cluster_stats[int(label)] = stats
    
    return cluster_stats

def generate_cluster_descriptions(cluster_stats, data, labels):
    """Generate human-readable cluster descriptions"""
    descriptions = {}
    
    for cluster_id, stats in cluster_stats.items():
        # Get the actual data for this cluster
        cluster_data = data[labels == cluster_id]
        
        # Get feature names from the actual data columns
        available_features = list(cluster_data.columns)
        
        # Calculate means for available features
        age_mean = cluster_data['Age'].mean() if 'Age' in available_features else 0
        income_mean = cluster_data['Annual_Income_Lakhs'].mean() if 'Annual_Income_Lakhs' in available_features else 0
        spending_mean = cluster_data['Spending_Score'].mean() if 'Spending_Score' in available_features else 0
        count = len(cluster_data)
        
        # Determine cluster characteristics based on available features
        if 'Annual_Income_Lakhs' in available_features and 'Spending_Score' in available_features:
            if income_mean > 15 and spending_mean > 80:
                segment_type = "Premium Customers"
                description = f"High-income customers with high spending scores - premium segment"
            elif income_mean < 10 and spending_mean < 70:
                segment_type = "Budget Customers"
                description = f"Low-income customers with low spending scores - budget segment"
            elif spending_mean > 80:
                segment_type = "High Spenders"
                description = f"High spending scores regardless of income - luxury seekers"
            elif 'Age' in available_features and age_mean < 35 and spending_mean > 75:
                segment_type = "Young Trend Followers"
                description = f"Young customers with high spending - trend followers"
            else:
                segment_type = "Mainstream Customers"
                description = f"Average income and spending - mainstream customers"
        else:
            segment_type = f"Cluster {cluster_id}"
            description = f"Customer segment with {count} members"
        
        descriptions[cluster_id] = {
            'segment_type': segment_type,
            'description': description,
            'count': count,
            'avg_age': age_mean,
            'avg_income': income_mean,
            'avg_spending': spending_mean
        }
    
    return descriptions

def create_download_link(data, filename, text):
    """Create download link for data"""
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def main():
    # Main header
    st.markdown('<h1 class="main-header">🎯 Interactive Customer Segmentation Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("Configuration Panel")
    
    # Dataset configuration
    st.sidebar.subheader("Dataset Configuration")
    n_customers = st.sidebar.slider("Number of Customers", 50, 1000, 200)
    age_min, age_max = st.sidebar.slider("Age Range", 18, 80, (20, 70))
    income_min, income_max = st.sidebar.slider("Annual Income Range (₹ Lakhs)", 2, 50, (3, 25))
    spending_min, spending_max = st.sidebar.slider("Spending Score Range", 1, 100, (50, 100))
    random_seed = st.sidebar.number_input("Random Seed", value=42, min_value=1, max_value=1000)
    
    # Clustering configuration
    st.sidebar.subheader("Clustering Configuration")
    features = st.sidebar.multiselect(
        "Select Features for Clustering",
        ['Age', 'Annual_Income_Lakhs', 'Spending_Score'],
        default=['Age', 'Annual_Income_Lakhs', 'Spending_Score']
    )
    
    auto_detect_k = st.sidebar.checkbox("Auto-detect Optimal K", value=True)
    custom_k = st.sidebar.slider("Custom K Value", 2, 8, 3, disabled=auto_detect_k)
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    show_stats = st.sidebar.checkbox("Show Statistical Summary", value=True)
    show_correlation = st.sidebar.checkbox("Show Correlation Matrix", value=True)
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
    
    # Generate or load data
    if st.sidebar.button("🔄 Generate New Data", type="primary"):
        st.cache_data.clear()
    
    # Generate sample data
    data = generate_sample_data(
        n_customers=n_customers,
        age_range=(age_min, age_max),
        income_range=(income_min, income_max),
        spending_range=(spending_min, spending_max),
        random_seed=random_seed
    )
    
    # Main content
    if len(features) < 2:
        st.error("⚠️ Please select at least 2 features for clustering analysis.")
        return
    
    # Data Overview
    st.subheader("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(data))
    with col2:
        st.metric("Features", len(features))
    with col3:
        st.metric("Age Range", f"{data['Age'].min():.0f} - {data['Age'].max():.0f}")
    with col4:
        st.metric("Income Range", f"₹{data['Annual_Income_Lakhs'].min():.1f} - ₹{data['Annual_Income_Lakhs'].max():.1f}L")
    
    # Statistical Summary
    if show_stats:
        st.subheader("📈 Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
    
    # Correlation Matrix
    if show_correlation:
        st.subheader("🔗 Feature Correlations")
        corr_fig = create_correlation_heatmap(data)
        st.plotly_chart(corr_fig, use_container_width=True)
    
    # Raw Data
    if show_raw_data:
        st.subheader("📋 Raw Data Preview")
        st.dataframe(data.head(10), use_container_width=True)
    
    # Clustering Analysis
    st.subheader("🎯 Clustering Analysis")
    
    if st.button("🚀 Run Clustering Analysis", type="primary"):
        with st.spinner("Running clustering analysis..."):
            # Prepare data for clustering
            clustering_data = data[features].copy()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(clustering_data)
            
            # Find optimal k and perform clustering
            if auto_detect_k:
                optimal_k, inertias, silhouette_scores, k_range = find_optimal_k(scaled_data)
                used_k = optimal_k
            else:
                used_k = custom_k
                optimal_k, inertias, silhouette_scores, k_range = find_optimal_k(scaled_data)
            
            # Perform clustering
            clustering_result = perform_clustering(scaled_data, used_k)
            labels = clustering_result['labels']
            centroids = clustering_result['centroids']
            
            # Display optimal k information
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Number of Clusters", used_k)
            with col2:
                st.metric("Silhouette Score", f"{silhouette_scores[used_k-2]:.3f}")
            with col3:
                st.metric("WCSS (Inertia)", f"{clustering_result['inertia']:.2f}")
            with col4:
                st.metric("Algorithm", "K-Means")
            
            # Elbow and Silhouette plots
            elbow_silhouette_fig = create_elbow_silhouette_plots(inertias, silhouette_scores, k_range)
            st.plotly_chart(elbow_silhouette_fig, use_container_width=True)
            
            # Cluster Visualization
            st.subheader("📈 Cluster Visualization")
            scatter_fig = create_cluster_scatter_plot(data, labels, centroids, features)
            if scatter_fig:
                st.plotly_chart(scatter_fig, use_container_width=True)
            
            # Cluster Distribution
            st.subheader("📊 Cluster Distribution")
            dist_fig = create_cluster_distribution_plots(data, labels, features)
            st.plotly_chart(dist_fig, use_container_width=True)
            
            # Cluster Statistics
            st.subheader("📋 Detailed Cluster Analysis")
            cluster_stats = calculate_cluster_statistics(data, labels, features)
            cluster_descriptions = generate_cluster_descriptions(cluster_stats, data, labels)
            
            # Display cluster information
            for cluster_id in sorted(cluster_stats.keys()):
                stats = cluster_stats[cluster_id]
                desc = cluster_descriptions[cluster_id]
                
                with st.expander(f"Cluster {cluster_id} - {desc['segment_type']} ({desc['count']} customers)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                         st.markdown(f"<div style='color: #FF0000; font-weight: bold; font-size: 16px;'>**Description:**</div>", unsafe_allow_html=True)
                         st.markdown(f"<div style='color: #FF0000; font-size: 14px; margin-bottom: 10px;'>{desc['description']}</div>", unsafe_allow_html=True)
                         if desc['avg_age'] > 0:
                             st.markdown(f"<div style='color: #FF0000; font-weight: bold; font-size: 16px;'>**Average Age:**</div>", unsafe_allow_html=True)
                             st.markdown(f"<div style='color: #FF0000; font-size: 14px; margin-bottom: 10px;'>{desc['avg_age']:.1f} years</div>", unsafe_allow_html=True)
                         if desc['avg_income'] > 0:
                             st.markdown(f"<div style='color: #FF0000; font-weight: bold; font-size: 16px;'>**Average Income:**</div>", unsafe_allow_html=True)
                             st.markdown(f"<div style='color: #FF0000; font-size: 14px; margin-bottom: 10px;'>₹{desc['avg_income']:.1f} Lakhs</div>", unsafe_allow_html=True)
                         if desc['avg_spending'] > 0:
                             st.markdown(f"<div style='color: #FF0000; font-weight: bold; font-size: 16px;'>**Average Spending Score:**</div>", unsafe_allow_html=True)
                             st.markdown(f"<div style='color: #FF0000; font-size: 14px; margin-bottom: 10px;'>{desc['avg_spending']:.1f}</div>", unsafe_allow_html=True)
                    
                    with col2:
                        # Create mini charts for this cluster
                        cluster_data = data[labels == cluster_id]
                        fig = make_subplots(rows=1, cols=len(features), subplot_titles=features)
                        
                        for i, feature in enumerate(features):
                            fig.add_trace(
                                go.Histogram(x=cluster_data[feature], name=feature, marker_color=CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]),
                                row=1, col=i+1
                            )
                        
                        fig.update_layout(
                            height=300, 
                            showlegend=False, 
                            plot_bgcolor='#f8f9fa', 
                            paper_bgcolor='#f8f9fa',
                            font=dict(color='#2c3e50', size=12)
                        )
                        
                        # Update axis labels for mini charts
                        for i in range(len(features)):
                            fig.update_xaxes(
                                title_text=features[i],
                                row=1, col=i+1,
                                title_font=dict(size=12, color='#2c3e50'),
                                tickfont=dict(size=10, color='#2c3e50')
                            )
                            fig.update_yaxes(
                                title_text="Count",
                                row=1, col=i+1,
                                title_font=dict(size=12, color='#2c3e50'),
                                tickfont=dict(size=10, color='#2c3e50')
                            )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Download Section
            st.subheader("💾 Download Results")
            
            # Add cluster labels to data
            data_with_clusters = data.copy()
            data_with_clusters['Cluster'] = labels
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(create_download_link(
                    data_with_clusters, 
                    f"customer_segmentation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "📊 Download Clustered Data"
                ), unsafe_allow_html=True)
            
            with col2:
                # Create cluster centers data
                centroids_df = pd.DataFrame(
                    centroids, 
                    columns=features,
                    index=[f'Cluster_{i}' for i in range(len(centroids))]
                )
                st.markdown(create_download_link(
                    centroids_df,
                    f"cluster_centers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "🎯 Download Cluster Centers"
                ), unsafe_allow_html=True)
            
            with col3:
                # Create cluster statistics summary
                stats_summary = []
                for cluster_id in sorted(cluster_stats.keys()):
                    desc = cluster_descriptions[cluster_id]
                    stats_summary.append({
                        'Cluster': cluster_id,
                        'Segment_Type': desc['segment_type'],
                        'Count': desc['count'],
                        'Description': desc['description']
                    })
                
                stats_df = pd.DataFrame(stats_summary)
                st.markdown(create_download_link(
                    stats_df,
                    f"cluster_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "📋 Download Cluster Summary"
                ), unsafe_allow_html=True)

if __name__ == "__main__":
    main() 