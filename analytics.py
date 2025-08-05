import pandas as pd
import numpy as np
from collections import defaultdict

class ClusterAnalytics:
    def __init__(self):
        self.segment_descriptions = {
            'high_value': "High-income customers with high spending scores - premium segment",
            'low_value': "Low-income customers with low spending scores - budget segment", 
            'high_spenders': "High spending scores regardless of income - luxury seekers",
            'moderate': "Average income and spending - mainstream customers",
            'young_high_spenders': "Young customers with high spending - trend followers"
        }
    
    def calculate_cluster_statistics(self, data, labels, feature_names):
        """Calculate comprehensive statistics for each cluster"""
        cluster_stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            cluster_data = data[mask]
            
            stats = {}
            for i, feature in enumerate(feature_names):
                feature_data = cluster_data[:, i]
                stats[str(feature)] = {
                    'mean': float(np.mean(feature_data)),
                    'median': float(np.median(feature_data)),
                    'std': float(np.std(feature_data)),
                    'min': float(np.min(feature_data)),
                    'max': float(np.max(feature_data)),
                    'count': int(len(feature_data))
                }
            
            cluster_stats[int(label)] = stats
        
        return cluster_stats
    
    def generate_cluster_descriptions(self, cluster_stats, feature_names):
        """Generate human-readable descriptions for each cluster"""
        descriptions = {}
        
        for cluster_id, stats in cluster_stats.items():
            # Extract key metrics
            age_mean = stats.get('Age', {}).get('mean', 0)
            income_mean = stats.get('Annual_Income_k', {}).get('mean', 0)
            spending_mean = stats.get('Spending_Score', {}).get('mean', 0)
            
            # Determine cluster characteristics
            age_category = self._categorize_age(age_mean)
            income_category = self._categorize_income(income_mean)
            spending_category = self._categorize_spending(spending_mean)
            
            # Generate description
            description = self._create_segment_description(
                cluster_id, age_category, income_category, spending_category,
                age_mean, income_mean, spending_mean
            )
            
            descriptions[int(cluster_id)] = {
                'description': description,
                'age_category': age_category,
                'income_category': income_category,
                'spending_category': spending_category,
                'key_metrics': {
                    'avg_age': round(age_mean, 1),
                    'avg_income': round(income_mean, 1),
                    'avg_spending_score': round(spending_mean, 1)
                }
            }
        
        return descriptions
    
    def _categorize_age(self, age):
        """Categorize age into segments"""
        if age < 30:
            return "Young"
        elif age < 45:
            return "Middle-aged"
        else:
            return "Senior"
    
    def _categorize_income(self, income):
        """Categorize income into segments"""
        if income < 40:
            return "Low"
        elif income < 70:
            return "Medium"
        else:
            return "High"
    
    def _categorize_spending(self, spending):
        """Categorize spending score into segments"""
        if spending < 30:
            return "Low"
        elif spending < 70:
            return "Medium"
        else:
            return "High"
    
    def _create_segment_description(self, cluster_id, age_cat, income_cat, spending_cat, 
                                  age_val, income_val, spending_val):
        """Create detailed description for a cluster"""
        
        # Define segment types based on characteristics
        if income_cat == "High" and spending_cat == "High":
            segment_type = "Premium Customers"
            business_value = "High-value customers requiring premium services and exclusive offers"
        elif income_cat == "Low" and spending_cat == "Low":
            segment_type = "Budget Customers"
            business_value = "Price-sensitive customers, focus on value deals and promotions"
        elif spending_cat == "High" and income_cat in ["Low", "Medium"]:
            segment_type = "High Spenders"
            business_value = "Luxury seekers, target with premium products despite income"
        elif age_cat == "Young" and spending_cat == "High":
            segment_type = "Young Trend Followers"
            business_value = "Early adopters, target with trendy products and social media marketing"
        else:
            segment_type = "Mainstream Customers"
            business_value = "Balanced customers, focus on variety and moderate pricing"
        
        description = f"""
        <strong>Cluster {cluster_id} - {segment_type}</strong><br>
        <strong>Demographics:</strong> {age_cat} customers (Average age: {age_val:.1f} years)<br>
        <strong>Financial Profile:</strong> {income_cat} income (${income_val:.1f}k/year), {spending_cat} spending tendency<br>
        <strong>Business Value:</strong> {business_value}<br>
        <strong>Marketing Strategy:</strong> {self._get_marketing_strategy(age_cat, income_cat, spending_cat)}
        """
        
        return description
    
    def _get_marketing_strategy(self, age_cat, income_cat, spending_cat):
        """Generate marketing strategy recommendations"""
        strategies = []
        
        if age_cat == "Young":
            strategies.append("Social media campaigns, mobile-first approach")
        elif age_cat == "Senior":
            strategies.append("Traditional marketing channels, clear communication")
        
        if income_cat == "High":
            strategies.append("Premium positioning, exclusive offers")
        elif income_cat == "Low":
            strategies.append("Value propositions, discounts and deals")
        
        if spending_cat == "High":
            strategies.append("Luxury products, VIP treatment")
        elif spending_cat == "Low":
            strategies.append("Budget-friendly options, cost-saving promotions")
        
        return "; ".join(strategies) if strategies else "Standard marketing approach"
    
    def calculate_business_metrics(self, cluster_stats, labels):
        """Calculate business-relevant metrics"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_customers = len(labels)
        
        business_metrics = {
            'total_customers': int(total_customers),
            'cluster_distribution': {},
            'revenue_potential': {},
            'customer_lifetime_value': {}
        }
        
        for i, label in enumerate(unique_labels):
            percentage = (counts[i] / total_customers) * 100
            business_metrics['cluster_distribution'][int(label)] = {
                'count': int(counts[i]),
                'percentage': round(percentage, 2)
            }
            
            # Calculate revenue potential (simplified)
            stats = cluster_stats[int(label)]
            avg_income = stats.get('Annual_Income_k', {}).get('mean', 0)
            avg_spending = stats.get('Spending_Score', {}).get('mean', 0)
            
            # Simple revenue potential calculation
            revenue_potential = (avg_income * avg_spending / 100) * counts[i]
            business_metrics['revenue_potential'][int(label)] = round(revenue_potential, 2)
            
            # Customer lifetime value (simplified)
            clv = avg_income * (avg_spending / 100) * 3  # Assuming 3-year relationship
            business_metrics['customer_lifetime_value'][int(label)] = round(clv, 2)
        
        return business_metrics
    
    def generate_recommendations(self, cluster_descriptions, business_metrics):
        """Generate actionable business recommendations"""
        recommendations = {
            'targeting_strategies': {},
            'product_recommendations': {},
            'pricing_strategies': {},
            'marketing_channels': {}
        }
        
        for cluster_id, description in cluster_descriptions.items():
            age_cat = description['age_category']
            income_cat = description['income_category']
            spending_cat = description['spending_category']
            
            # Targeting strategies
            if income_cat == "High" and spending_cat == "High":
                recommendations['targeting_strategies'][int(cluster_id)] = "Premium targeting with exclusive access"
                recommendations['product_recommendations'][int(cluster_id)] = "Luxury products, premium services"
                recommendations['pricing_strategies'][int(cluster_id)] = "Premium pricing, value-based pricing"
                recommendations['marketing_channels'][int(cluster_id)] = "Direct marketing, exclusive events"
            
            elif income_cat == "Low" and spending_cat == "Low":
                recommendations['targeting_strategies'][int(cluster_id)] = "Mass market targeting"
                recommendations['product_recommendations'][int(cluster_id)] = "Budget-friendly products, basic services"
                recommendations['pricing_strategies'][int(cluster_id)] = "Competitive pricing, volume discounts"
                recommendations['marketing_channels'][int(cluster_id)] = "Social media, email campaigns"
            
            elif spending_cat == "High":
                recommendations['targeting_strategies'][int(cluster_id)] = "Aspirational targeting"
                recommendations['product_recommendations'][int(cluster_id)] = "Trendy products, limited editions"
                recommendations['pricing_strategies'][int(cluster_id)] = "Value-based pricing, installment options"
                recommendations['marketing_channels'][int(cluster_id)] = "Influencer marketing, social media"
            
            else:
                recommendations['targeting_strategies'][int(cluster_id)] = "Balanced targeting approach"
                recommendations['product_recommendations'][int(cluster_id)] = "Variety of products, standard services"
                recommendations['pricing_strategies'][int(cluster_id)] = "Competitive pricing, loyalty programs"
                recommendations['marketing_channels'][int(cluster_id)] = "Multi-channel approach"
        
        return recommendations
    
    def create_summary_report(self, cluster_stats, cluster_descriptions, business_metrics):
        """Create a comprehensive summary report"""
        report = {
            'executive_summary': self._create_executive_summary(business_metrics),
            'cluster_insights': cluster_descriptions,
            'business_metrics': business_metrics,
            'key_findings': self._extract_key_findings(cluster_stats, business_metrics)
        }
        
        return report
    
    def _create_executive_summary(self, business_metrics):
        """Create executive summary"""
        total_customers = business_metrics['total_customers']
        n_clusters = len(business_metrics['cluster_distribution'])
        
        # Find largest cluster
        largest_cluster = max(business_metrics['cluster_distribution'].items(), 
                            key=lambda x: x[1]['count'])
        
        # Find highest revenue potential cluster
        highest_revenue = max(business_metrics['revenue_potential'].items(), 
                            key=lambda x: x[1])
        
        summary = f"""
        <strong>Executive Summary</strong><br>
        • Total customers analyzed: {total_customers}<br>
        • Number of customer segments identified: {n_clusters}<br>
        • Largest segment: Cluster {largest_cluster[0]} ({largest_cluster[1]['percentage']}% of customers)<br>
        • Highest revenue potential: Cluster {highest_revenue[0]} (${highest_revenue[1]:,.0f})<br>
        • Average customers per segment: {total_customers // n_clusters}
        """
        
        return summary
    
    def _extract_key_findings(self, cluster_stats, business_metrics):
        """Extract key insights from the analysis"""
        findings = []
        
        # Find most diverse cluster
        cluster_diversities = {}
        for cluster_id, stats in cluster_stats.items():
            age_std = stats.get('Age', {}).get('std', 0)
            income_std = stats.get('Annual_Income_k', {}).get('std', 0)
            spending_std = stats.get('Spending_Score', {}).get('std', 0)
            
            # Calculate diversity score (higher std = more diverse)
            diversity = (age_std + income_std + spending_std) / 3
            cluster_diversities[cluster_id] = diversity
        
        most_diverse = max(cluster_diversities.items(), key=lambda x: x[1])
        findings.append(f"Cluster {most_diverse[0]} shows the highest internal diversity")
        
        # Find most homogeneous cluster
        most_homogeneous = min(cluster_diversities.items(), key=lambda x: x[1])
        findings.append(f"Cluster {most_homogeneous[0]} is the most homogeneous segment")
        
        # Revenue insights
        total_revenue = sum(business_metrics['revenue_potential'].values())
        findings.append(f"Total revenue potential across all segments: ${total_revenue:,.0f}")
        
        return findings 