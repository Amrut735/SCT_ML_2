// Customer Segmentation Application JavaScript

class CustomerSegmentationApp {
    constructor() {
        this.initializeEventListeners();
        this.currentData = null;
        this.analysisResults = null;
    }

    initializeEventListeners() {
        // File upload form
        document.getElementById('uploadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleFileUpload();
        });

        // Sample data button
        document.getElementById('useSampleData').addEventListener('click', () => {
            this.loadSampleData();
        });

        // Run analysis button
        document.getElementById('runAnalysis').addEventListener('click', () => {
            this.runAnalysis();
        });

        // Download results button
        document.getElementById('downloadResults').addEventListener('click', () => {
            this.downloadResults();
        });
    }

    async handleFileUpload() {
        const fileInput = document.getElementById('csvFile');
        const file = fileInput.files[0];

        if (!file) {
            this.showError('Please select a file to upload.');
            return;
        }

        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showError('Please upload a CSV file.');
            return;
        }

        this.showLoading();
        this.hideError();

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                this.currentData = result.data_summary;
                this.showSuccess('File uploaded successfully!');
                this.displayDataPreview();
                this.showAnalysisConfig();
            } else {
                this.showError(result.error || 'Upload failed.');
            }
        } catch (error) {
            this.showError('Upload failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    async loadSampleData() {
        this.showLoading();
        this.hideError();

        try {
            const response = await fetch('/use_sample_data', {
                method: 'POST'
            });

            const result = await response.json();

            if (result.success) {
                this.currentData = result.data_summary;
                this.showSuccess('Sample data loaded successfully!');
                this.displayDataPreview();
                this.showAnalysisConfig();
            } else {
                this.showError(result.error || 'Failed to load sample data.');
            }
        } catch (error) {
            this.showError('Failed to load sample data: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayDataPreview() {
        if (!this.currentData) return;

        // Display data summary
        const summaryHtml = this.createDataSummaryHTML();
        document.getElementById('dataSummary').innerHTML = summaryHtml;

        // Load data preview table
        this.loadDataTable();

        // Show the preview section
        document.getElementById('dataPreviewSection').style.display = 'block';
    }

    createDataSummaryHTML() {
        const summary = this.currentData;
        return `
            <div class="data-summary-grid">
                <div class="summary-item">
                    <div class="summary-value">${summary.total_customers}</div>
                    <div class="summary-label">Total Customers</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">${summary.features.length}</div>
                    <div class="summary-label">Features</div>
                </div>
                <div class="summary-item">
                    <div class="summary-value">${Object.values(summary.missing_values).reduce((a, b) => a + b, 0)}</div>
                    <div class="summary-label">Missing Values</div>
                </div>
            </div>
        `;
    }

    async loadDataTable() {
        try {
            const response = await fetch('/get_data_preview');
            const result = await response.json();

            if (result.preview) {
                const tableHtml = this.createDataTableHTML(result.preview, result.columns);
                document.getElementById('dataTable').innerHTML = tableHtml;
            }
        } catch (error) {
            console.error('Failed to load data table:', error);
        }
    }

    createDataTableHTML(data, columns) {
        let tableHtml = `
            <div class="table-responsive">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            ${columns.map(col => `<th>${col}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
        `;

        data.forEach(row => {
            tableHtml += '<tr>';
            columns.forEach(col => {
                tableHtml += `<td>${row[col] || ''}</td>`;
            });
            tableHtml += '</tr>';
        });

        tableHtml += `
                    </tbody>
                </table>
            </div>
        `;

        return tableHtml;
    }

    showAnalysisConfig() {
        // Populate feature selection
        this.populateFeatureSelection();
        
        // Show the configuration section
        document.getElementById('analysisConfigSection').style.display = 'block';
    }

    populateFeatureSelection() {
        if (!this.currentData) return;

        const featureSelection = document.getElementById('featureSelection');
        const defaultFeatures = ['Age', 'Annual_Income_k', 'Spending_Score'];
        
        let html = '';
        this.currentData.features.forEach(feature => {
            const isChecked = defaultFeatures.includes(feature) ? 'checked' : '';
            html += `
                <div class="feature-checkbox">
                    <input type="checkbox" id="feature_${feature}" value="${feature}" ${isChecked}>
                    <label for="feature_${feature}">${feature}</label>
                </div>
            `;
        });
        
        featureSelection.innerHTML = html;
    }

    async runAnalysis() {
        this.showLoading();
        this.hideError();

        // Get selected features
        const selectedFeatures = Array.from(document.querySelectorAll('#featureSelection input:checked'))
            .map(input => input.value);

        if (selectedFeatures.length === 0) {
            this.showError('Please select at least one feature for analysis.');
            this.hideLoading();
            return;
        }

        // Get custom k value
        const customK = document.getElementById('customK').value;
        const analysisParams = {
            features: selectedFeatures,
            custom_k: customK ? parseInt(customK) : null
        };

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(analysisParams)
            });

            const result = await response.json();

            if (result.success) {
                this.analysisResults = result;
                this.displayResults();
                this.showSuccess('Analysis completed successfully!');
            } else {
                this.showError(result.error || 'Analysis failed.');
            }
        } catch (error) {
            this.showError('Analysis failed: ' + error.message);
        } finally {
            this.hideLoading();
        }
    }

    displayResults() {
        if (!this.analysisResults) return;

        // Display executive summary
        this.displayExecutiveSummary();

        // Display visualizations
        this.displayVisualizations();

        // Display cluster analysis
        this.displayClusterAnalysis();

        // Display business metrics
        this.displayBusinessMetrics();

        // Show results section
        document.getElementById('resultsSection').style.display = 'block';
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    displayExecutiveSummary() {
        const summary = this.analysisResults.summary_report.executive_summary;
        document.getElementById('executiveSummary').innerHTML = summary;
    }

    displayVisualizations() {
        const viz = this.analysisResults.visualizations;

        // Set images
        if (viz.elbow_plot) {
            document.getElementById('elbowPlot').src = 'data:image/png;base64,' + viz.elbow_plot;
        }
        if (viz.scatter_plot) {
            document.getElementById('scatterPlot').src = 'data:image/png;base64,' + viz.scatter_plot;
        }
        if (viz.distribution_plot) {
            document.getElementById('distributionPlot').src = 'data:image/png;base64,' + viz.distribution_plot;
        }
        if (viz.heatmap) {
            document.getElementById('heatmapPlot').src = 'data:image/png;base64,' + viz.heatmap;
        }
        if (viz.size_chart) {
            document.getElementById('sizeChart').src = 'data:image/png;base64,' + viz.size_chart;
        }
    }

    displayClusterAnalysis() {
        const descriptions = this.analysisResults.cluster_descriptions;
        const stats = this.analysisResults.cluster_stats;

        let html = '<div class="cluster-grid">';
        
        Object.keys(descriptions).forEach(clusterId => {
            const description = descriptions[clusterId];
            const clusterStats = stats[clusterId];
            
            html += `
                <div class="cluster-card">
                    <div class="cluster-header">
                        <div class="cluster-title">Cluster ${clusterId}</div>
                        <div class="cluster-size">${clusterStats.Age?.count || 0} customers</div>
                    </div>
                    <div class="cluster-description">
                        ${description.description}
                    </div>
                    <div class="cluster-metrics mt-3">
                        <div class="row">
                            <div class="col-4">
                                <div class="text-center">
                                    <div class="metric-value">${description.key_metrics.avg_age}</div>
                                    <div class="metric-label">Avg Age</div>
                                </div>
                            </div>
                            <div class="col-4">
                                <div class="text-center">
                                    <div class="metric-value">$${description.key_metrics.avg_income}k</div>
                                    <div class="metric-label">Avg Income</div>
                                </div>
                            </div>
                            <div class="col-4">
                                <div class="text-center">
                                    <div class="metric-value">${description.key_metrics.avg_spending_score}</div>
                                    <div class="metric-label">Spending Score</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        html += '</div>';
        document.getElementById('clusterAnalysis').innerHTML = html;
    }

    displayBusinessMetrics() {
        const metrics = this.analysisResults.business_metrics;
        
        let html = '<div class="metrics-grid">';
        
        // Total customers
        html += `
            <div class="metric-card">
                <div class="metric-value">${metrics.total_customers}</div>
                <div class="metric-label">Total Customers</div>
            </div>
        `;

        // Cluster distribution
        Object.keys(metrics.cluster_distribution).forEach(clusterId => {
            const dist = metrics.cluster_distribution[clusterId];
            html += `
                <div class="metric-card">
                    <div class="metric-value">${dist.count}</div>
                    <div class="metric-label">Cluster ${clusterId} (${dist.percentage}%)</div>
                </div>
            `;
        });

        // Revenue potential
        Object.keys(metrics.revenue_potential).forEach(clusterId => {
            const revenue = metrics.revenue_potential[clusterId];
            html += `
                <div class="metric-card">
                    <div class="metric-value">$${revenue.toLocaleString()}</div>
                    <div class="metric-label">Cluster ${clusterId} Revenue</div>
                </div>
            `;
        });
        
        html += '</div>';
        document.getElementById('businessMetrics').innerHTML = html;
    }

    async downloadResults() {
        try {
            const response = await fetch('/download_results', {
                method: 'POST'
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'customer_segmentation_results.csv';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showSuccess('Results downloaded successfully!');
            } else {
                const result = await response.json();
                this.showError(result.error || 'Download failed.');
            }
        } catch (error) {
            this.showError('Download failed: ' + error.message);
        }
    }

    showLoading() {
        document.getElementById('loadingSection').style.display = 'block';
    }

    hideLoading() {
        document.getElementById('loadingSection').style.display = 'none';
    }

    showError(message) {
        const errorSection = document.getElementById('errorSection');
        const errorMessage = document.getElementById('errorMessage');
        errorMessage.textContent = message;
        errorSection.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            this.hideError();
        }, 5000);
    }

    hideError() {
        document.getElementById('errorSection').style.display = 'none';
    }

    showSuccess(message) {
        // Create a temporary success alert
        const alertHtml = `
            <div class="alert alert-success alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        // Insert at the top of the container
        const container = document.querySelector('.container');
        container.insertAdjacentHTML('afterbegin', alertHtml);
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            const alert = container.querySelector('.alert-success');
            if (alert) {
                alert.remove();
            }
        }, 3000);
    }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CustomerSegmentationApp();
}); 