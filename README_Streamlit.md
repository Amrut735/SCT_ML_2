# 🎯 Interactive Customer Segmentation Dashboard - Streamlit

A comprehensive, interactive customer segmentation dashboard built with Streamlit that provides advanced K-Means clustering analysis with beautiful visualizations and real-time insights.

## 🌟 Features

### 🎛️ **Interactive Configuration Panel**
- **Dataset Configuration**: Customize number of customers, age range, income range, spending score range
- **Clustering Configuration**: Select features for clustering, auto-detect optimal K or set custom K
- **Analysis Options**: Toggle statistical summary, correlation matrix, and raw data display
- **Random Seed Control**: Reproducible results with seed control

### 📊 **Advanced Analytics**
- **K-Means Clustering**: Automatic optimal K detection using Elbow method and Silhouette analysis
- **Feature Correlation Matrix**: Interactive heatmap showing feature relationships
- **Statistical Summary**: Comprehensive descriptive statistics
- **Real-time Data Generation**: Generate custom datasets on-the-fly

### 📈 **Interactive Visualizations**
- **Elbow & Silhouette Plots**: Side-by-side analysis for optimal cluster determination
- **Cluster Scatter Plot**: Interactive scatter plot with cluster centers marked
- **Cluster Distribution**: Pie chart and bar chart showing customer distribution
- **Feature Histograms**: Individual cluster feature distributions

### 🎯 **Business Intelligence**
- **Automatic Segment Classification**: Premium, Budget, High Spenders, Young Trend Followers, Mainstream
- **Detailed Cluster Analysis**: Expandable sections with comprehensive statistics
- **Customer Insights**: Average age, income, and spending score per segment
- **Business Recommendations**: Automatic segment descriptions and characteristics

### 💾 **Export Capabilities**
- **Download Clustered Data**: Complete dataset with cluster labels
- **Download Cluster Centers**: Centroid coordinates for each cluster
- **Download Analysis Report**: Comprehensive business report with segment details

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd SCT_ML_2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Run the Streamlit application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📖 How to Use

### 1. **Configuration Setup**
   - **Dataset Configuration**: Adjust sliders for customer count, age range, income range, and spending score range
   - **Clustering Configuration**: Select features for clustering (Age, Annual Income, Spending Score)
   - **Analysis Options**: Choose which sections to display

### 2. **Generate Analysis**
   - Click the "🚀 Generate Analysis" button
   - The application will generate sample data and perform clustering analysis
   - View real-time results and visualizations

### 3. **Explore Results**
   - **Key Metrics**: View customer count, average age, income, and spending score
   - **Statistical Summary**: Comprehensive data statistics
   - **Correlation Matrix**: Feature relationship heatmap
   - **Clustering Analysis**: Elbow and Silhouette plots for optimal K determination

### 4. **Cluster Analysis**
   - **Cluster Visualization**: Interactive scatter plot with cluster centers
   - **Cluster Distribution**: Pie chart and bar chart showing customer distribution
   - **Detailed Analysis**: Expandable sections for each cluster with statistics and histograms

### 5. **Export Results**
   - **Download Clustered Data**: Get the complete dataset with cluster labels
   - **Download Cluster Centers**: Export centroid coordinates
   - **Download Analysis Report**: Comprehensive business report

## 🌐 Deployment Options

### Streamlit Cloud (Recommended)
1. **Push to GitHub**: Upload your code to a GitHub repository
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Select your repository
   - Set the main file path to `streamlit_app.py`
3. **Deploy**: Click "Deploy" and get your public URL

### Heroku
1. **Create Procfile**:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. **Create requirements.txt**:
   ```bash
   pip freeze > requirements.txt
   ```
3. **Deploy to Heroku**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Local Network
```bash
streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

## 📊 Sample Data Format

The application generates sample data with the following structure:

```csv
CustomerID,Age,Annual_Income_Lakhs,Spending_Score
1,45.2,13.45,78.3
2,38.7,16.23,82.1
3,52.1,11.89,65.4
...
```

### Features:
- **CustomerID**: Unique customer identifier
- **Age**: Customer age in years
- **Annual_Income_Lakhs**: Annual income in Indian Rupees (Lakhs)
- **Spending_Score**: Spending score from 1-100

## 🔧 Technical Details

### Algorithm Implementation
- **K-Means Clustering**: Scikit-learn implementation with n_init=10, max_iter=300
- **Optimal K Detection**: 
  - Elbow method for inertia analysis
  - Silhouette score for cluster quality
  - Combined approach for robust selection
- **Data Scaling**: StandardScaler for feature normalization
- **Performance Constraints**: Optimized for datasets up to 1000 customers

### Visualization Technologies
- **Plotly**: Interactive charts and plots
- **Streamlit**: Web interface and data display
- **Custom CSS**: Beautiful styling and responsive design

### Business Intelligence Features
- **Automatic Segment Classification**: Based on income and spending patterns
- **Customer Insights**: Demographic and behavioral analysis
- **Export Functionality**: Multiple download options for business use

## 🎨 Customization

### Styling
Modify the CSS in the `st.markdown()` section to customize:
- Colors and gradients
- Card layouts
- Typography
- Spacing and margins

### Features
Add new features by:
- Extending the sidebar configuration
- Adding new visualization functions
- Creating additional export formats
- Implementing new clustering algorithms

### Data Sources
Replace the sample data generation with:
- CSV file upload
- Database connections
- API integrations
- Real-time data streams

## 📈 Business Applications

### Marketing Strategy
- **Targeted Campaigns**: Identify high-value customer segments
- **Personalized Offers**: Tailor promotions based on segment characteristics
- **Customer Retention**: Focus on at-risk segments

### Product Development
- **Feature Prioritization**: Understand customer needs by segment
- **Pricing Strategy**: Optimize pricing for different segments
- **Market Expansion**: Identify growth opportunities

### Customer Service
- **Service Level Optimization**: Allocate resources based on segment value
- **Communication Strategy**: Customize messaging for different segments
- **Loyalty Programs**: Design programs for specific customer types

## 🔍 Troubleshooting

### Common Issues

1. **Installation Errors**
   ```bash
   pip install --upgrade pip
   pip install -r requirements_streamlit.txt
   ```

2. **Port Already in Use**
   ```bash
   streamlit run streamlit_app.py --server.port=8502
   ```

3. **Memory Issues**
   - Reduce the number of customers in the configuration
   - Use fewer features for clustering
   - Increase system memory

4. **Display Issues**
   - Clear browser cache
   - Try different browsers
   - Check Streamlit version compatibility

### Performance Optimization
- **Large Datasets**: Consider sampling for initial analysis
- **Feature Selection**: Use only relevant features
- **Caching**: Results are cached for faster access

## 📚 Dependencies

### Core Libraries
- **Streamlit 1.28.0+**: Web application framework
- **Pandas 1.5.0+**: Data manipulation and analysis
- **NumPy 1.21.0+**: Numerical computing
- **Scikit-learn 1.0.0+**: Machine learning algorithms
- **Plotly 5.0.0+**: Interactive visualizations
- **Seaborn 0.11.0+**: Statistical visualizations
- **Matplotlib 3.5.0+**: Plotting library

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Create an issue with detailed information

## 🎉 Acknowledgments

- Built with Streamlit and Scikit-learn
- Inspired by retail customer segmentation needs
- Designed for educational and business use
- Interactive visualizations powered by Plotly

---

**Happy Clustering! 🎯📊**

*Transform your customer data into actionable business insights with this comprehensive segmentation dashboard.* 