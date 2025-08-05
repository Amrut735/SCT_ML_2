# 🎯 Interactive Customer Segmentation Dashboard

A powerful and interactive Streamlit application for customer segmentation using K-Means clustering with beautiful visualizations and real-time analysis.

## 🌐 **Live Demo**

**[🚀 Try the Live Application](https://kmeans-customerlens.streamlit.app/)**

Experience the full interactive dashboard with real-time clustering analysis and beautiful visualizations.

## 🌟 Features

### 📊 **Interactive Dashboard**
- **Real-time Data Generation**: Generate synthetic customer data with customizable parameters
- **Dynamic Clustering**: Automatic optimal cluster detection using Elbow method and Silhouette analysis
- **Beautiful Visualizations**: Colorful charts and graphs with modern styling

### 🎨 **Visual Design**
- **Gradient Backgrounds**: Beautiful purple-blue gradient expanders
- **Colorful Theme**: Vibrant color palette with red text for cluster analysis
- **Responsive Layout**: Wide layout with sidebar configuration panel
- **Hover Effects**: Interactive animations and smooth transitions

### 📈 **Analytics & Insights**
- **Correlation Analysis**: Feature correlation heatmap with Red-Blue color scale
- **Cluster Visualization**: Interactive scatter plots with cluster centers
- **Distribution Analysis**: Feature distribution by cluster with histograms
- **Statistical Summary**: Comprehensive cluster statistics and descriptions

### 🔧 **Configuration Options**
- **Dataset Parameters**: Customize number of customers, age/income ranges
- **Clustering Settings**: Auto-detect optimal K or set custom values
- **Feature Selection**: Choose which features to include in clustering
- **Analysis Options**: Toggle various analysis components

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-dashboard.git
   cd customer-segmentation-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Access the dashboard**
   - Open your browser and go to `http://localhost:8501`
   - The application will automatically open in your default browser

## 📋 Requirements

The application uses the following key libraries:
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning algorithms (K-Means clustering)
- **Pandas & NumPy**: Data manipulation and analysis
- **Seaborn**: Statistical data visualization

## 🎯 How to Use

### 1. **Configure Your Data**
   - Use the sidebar to adjust dataset parameters
   - Set number of customers, age range, income range, and spending score range
   - Click "Generate New Data" to create fresh synthetic data

### 2. **Run Clustering Analysis**
   - Select features for clustering (Age, Income, Spending Score)
   - Choose between auto-detect optimal K or custom K value
   - Click "Run Clustering Analysis" to start the process

### 3. **Explore Results**
   - View correlation matrix to understand feature relationships
   - Examine elbow and silhouette plots for optimal cluster validation
   - Explore cluster visualizations and distributions
   - Read detailed cluster analysis with customer segment descriptions

### 4. **Download Results**
   - Download clustered data as CSV
   - Export cluster centers and summary statistics
   - Save analysis results for further processing

## 🎨 Customization

### **Color Themes**
The application uses a vibrant color palette:
- **Primary Colors**: Red (`#FF6B6B`), Teal (`#4ECDC4`)
- **Gradient Backgrounds**: Purple-blue gradients for expanders
- **Cluster Colors**: 10 different colors for cluster visualization

### **Styling**
- **CSS Customization**: All styling is defined in the main application file
- **Responsive Design**: Adapts to different screen sizes
- **Interactive Elements**: Hover effects and smooth transitions

## 📊 Sample Output

The application generates:
- **3-5 Customer Segments**: Automatically detected based on data characteristics
- **Segment Types**: Premium Customers, High Spenders, Young Trend Followers, Mainstream Customers, Budget Customers
- **Statistical Insights**: Average age, income, and spending scores for each segment
- **Visual Charts**: Distribution plots, correlation matrices, and cluster visualizations

## 🔧 Technical Details

### **Clustering Algorithm**
- **Algorithm**: K-Means clustering with standardized features
- **Optimization**: Elbow method and Silhouette analysis for optimal K
- **Preprocessing**: StandardScaler for feature normalization
- **Validation**: Multiple metrics for cluster quality assessment

### **Data Generation**
- **Synthetic Data**: Realistic customer data with correlations
- **Features**: Age, Annual Income (₹ Lakhs), Spending Score
- **Correlations**: Age-income correlation, age-spending inverse correlation
- **Customizable**: All parameters adjustable through the interface

## 📁 Project Structure

```
customer-segmentation-dashboard/
├── streamlit_app.py          # Main application file
├── requirements_streamlit.txt # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore rules
├── static/                  # Static assets (if any)
└── uploads/                 # Upload directory (gitignored)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web framework
- **Plotly** for interactive visualizations
- **Scikit-learn** for machine learning algorithms
- **Pandas & NumPy** for data manipulation

## 📞 Support

If you have any questions or need help with the application:
- Create an issue on GitHub
- Check the documentation in the code comments
- Review the Streamlit documentation for additional features

---

**Made with ❤️ using Streamlit, Plotly, and Scikit-learn** 