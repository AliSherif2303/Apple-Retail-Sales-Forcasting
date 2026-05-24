import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. LOAD AND PREPROCESS DATA
# ==========================================

def load_csv_safe(file_path):
    """Load CSV with multiple encoding attempts."""
    for enc in ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise Exception(f"Could not decode {file_path}")

print("="*80)
print("PROFESSIONAL MARKET EXPANSION INTELLIGENCE SYSTEM")
print("="*80)

# Load data
print("\n📂 Loading data...")
sales_df = load_csv_safe('../data/processed/merged_city_sales_data.csv')
country_df = load_csv_safe('../data/processed/merged_data_with_population.csv')

print(f"✓ Loaded {len(sales_df):,} transactions")
print(f"✓ Loaded {len(country_df)} potential markets")

# ==========================================
# 2. AGGREGATE HISTORICAL PERFORMANCE
# ==========================================
print("\n" + "="*80)
print("STEP 1: Building Historical Market Intelligence")
print("="*80)

# Clean and prepare historical data
historical_raw = sales_df.copy()

# Ensure numeric columns
numeric_cols = ['sales_amount_realistic', 'gdp_per_capita', 'Population', 
                'internet_usage_pct', 'inflation_rate', 'exchange_rate']
for col in numeric_cols:
    if col in historical_raw.columns:
        historical_raw[col] = pd.to_numeric(historical_raw[col], errors='coerce')

# Aggregate by country
historical = historical_raw.groupby('country_norm_mapped').agg({
    'sales_amount_realistic': ['sum', 'count', 'mean'],
    'gdp_per_capita': 'mean',
    'Population': 'mean',
    'internet_usage_pct': 'mean',
    'inflation_rate': 'mean',
    'exchange_rate': 'mean'
}).round(2)

# Flatten column names
historical.columns = ['TotalSales', 'TransactionCount', 'AvgTransactionValue',
                      'GDP', 'Population', 'InternetUsage', 'Inflation', 'ExchangeRate']
historical = historical.reset_index()
historical.rename(columns={'country_norm_mapped': 'Country'}, inplace=True)

# Calculate key metrics
historical['SalesPerCapita'] = historical['TotalSales'] / historical['Population']
historical['SalesPerGDP'] = historical['TotalSales'] / historical['GDP']
historical['MarketPenetration'] = (historical['TransactionCount'] / historical['Population']) * 100

# Sort by performance
historical = historical.sort_values('SalesPerCapita', ascending=False)

print(f"\n📊 Historical Market Summary:")
print(f"   Total markets in history: {len(historical)}")
print(f"   Total sales value: ${historical['TotalSales'].sum():,.0f}")
print(f"   Average sales per capita: ${historical['SalesPerCapita'].mean():.2f}")
print(f"   Best performer: {historical.iloc[0]['Country']} (${historical.iloc[0]['SalesPerCapita']:.2f}/person)")
print(f"   Worst performer: {historical.iloc[-1]['Country']} (${historical.iloc[-1]['SalesPerCapita']:.2f}/person)")

# Display top performers
print("\n🏆 Top 5 Historical Markets:")
for i, row in historical.head(5).iterrows():
    print(f"   {i+1}. {row['Country']}: ${row['SalesPerCapita']:.2f}/person | "
          f"GDP: ${row['GDP']:,.0f} | Internet: {row['InternetUsage']:.0f}%")

# ==========================================
# 3. DEFINE WEIGHTED SCORING SYSTEM
# ==========================================
print("\n" + "="*80)
print("STEP 2: Weighted Scoring System Definition")
print("="*80)

# Calculate dynamic thresholds from historical data
gdp_percentiles = historical['GDP'].quantile([0.25, 0.5, 0.75])
pop_percentiles = historical['Population'].quantile([0.25, 0.5, 0.75])
internet_percentiles = historical['InternetUsage'].quantile([0.25, 0.5, 0.75])
inflation_percentiles = historical['Inflation'].quantile([0.25, 0.5, 0.75])

# Define weights (total = 100%)
WEIGHTS = {
    'gdp': 0.35,      # Economic strength
    'population': 0.20,  # Market size
    'internet': 0.15,    # Digital readiness
    'inflation': 0.15,   # Economic stability
    'exchange': 0.05,    # Currency risk
    'historical': 0.10   # Similar market performance
}

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│  WEIGHTED SCORING SYSTEM                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Factor              Weight    Why it matters                               │
│  ─────────────────────────────────────────────────────────────────────      │
│  GDP per capita      35%       Purchasing power & price sensitivity         │
│  Population          20%       Total addressable market size                │
│  Internet usage      15%       Digital marketing & e-commerce readiness     │
│  Inflation rate      15%       Economic stability & pricing power           │
│  Exchange stability   5%       Currency risk & profit repatriation          │
│  Historical similarity 10%     Proven success in similar conditions         │
│                                                                              │
│  TOTAL:              100%                                                   │
│                                                                              │
│  Score Range: 0-100 (higher = better opportunity)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# ==========================================
# 4. PREPARE TARGET MARKETS
# ==========================================
print("\n" + "="*80)
print("STEP 3: Preparing Target Market Data")
print("="*80)

# Prepare target countries data
targets = country_df.copy()
targets.rename(columns={
    'Country Name': 'Country',
    'GDP per capita (USD)': 'GDP',
    'Population': 'Population',
    'Internet Usage (%)': 'Internet',
    'Inflation Rate (%)': 'Inflation',
    'Official Exchange Rate': 'ExchangeRate'
}, inplace=True)

# Ensure numeric
targets['Population'] = pd.to_numeric(targets['Population'], errors='coerce')
targets['GDP'] = pd.to_numeric(targets['GDP'], errors='coerce')
targets['Internet'] = pd.to_numeric(targets['Internet'], errors='coerce')
targets['Inflation'] = pd.to_numeric(targets['Inflation'], errors='coerce')
targets['ExchangeRate'] = pd.to_numeric(targets['ExchangeRate'], errors='coerce')

# Drop rows with critical missing data
targets = targets.dropna(subset=['Country', 'GDP', 'Population'])
print(f"✓ {len(targets)} target markets ready for evaluation")

# ==========================================
# 5. MULTI-FACTOR COUNTRY SIMILARITY
# ==========================================

def calculate_similarity_score(target_row, historical_df):
    """
    Calculate similarity between target country and best historical performers
    using multiple factors: GDP, Internet, Inflation, Population
    """
    similarities = []
    
    for _, hist_row in historical_df.iterrows():
        # Calculate similarity for each factor (closer to 1 = more similar)
        gdp_sim = 1 - min(1, abs(target_row['GDP'] - hist_row['GDP']) / max(target_row['GDP'], hist_row['GDP']))
        internet_sim = 1 - min(1, abs(target_row['Internet'] - hist_row['InternetUsage']) / 100)
        inflation_sim = 1 - min(1, abs(target_row['Inflation'] - hist_row['Inflation']) / 20)
        pop_sim = 1 - min(1, abs(np.log10(target_row['Population']) - np.log10(hist_row['Population'])) / 2)
        
        # Weighted average similarity
        total_sim = (gdp_sim * 0.4 + internet_sim * 0.3 + 
                     inflation_sim * 0.2 + pop_sim * 0.1)
        
        # Weight by historical performance
        weighted_sim = total_sim * (hist_row['SalesPerCapita'] / historical_df['SalesPerCapita'].max())
        similarities.append(weighted_sim)
    
    # Return max similarity (best match) and average of top 3
    similarities = np.array(similarities)
    return {
        'best_match': similarities.max(),
        'top3_avg': np.mean(sorted(similarities, reverse=True)[:3]),
        'overall': np.mean(similarities)
    }

print("\n📊 Calculating market similarities...")
similarity_results = []
for idx, row in targets.iterrows():
    sim = calculate_similarity_score(row, historical)
    similarity_results.append(sim)

targets['Similarity_Best'] = [s['best_match'] for s in similarity_results]
targets['Similarity_Top3'] = [s['top3_avg'] for s in similarity_results]
targets['Similarity_Overall'] = [s['overall'] for s in similarity_results]

# ==========================================
# 6. WEIGHTED SCORING CALCULATION
# ==========================================

def calculate_weighted_score(row):
    """Calculate weighted score (0-100) for each market"""
    score = 0
    
    # 1. GDP Score (0-100)
    if row['GDP'] >= gdp_percentiles[0.75]:
        gdp_score = 100
    elif row['GDP'] >= gdp_percentiles[0.5]:
        gdp_score = 70
    elif row['GDP'] >= gdp_percentiles[0.25]:
        gdp_score = 40
    else:
        gdp_score = 20
    score += gdp_score * WEIGHTS['gdp']
    
    # 2. Population Score (0-100)
    if row['Population'] >= pop_percentiles[0.75]:
        pop_score = 100
    elif row['Population'] >= pop_percentiles[0.5]:
        pop_score = 70
    elif row['Population'] >= pop_percentiles[0.25]:
        pop_score = 40
    else:
        pop_score = 20
    score += pop_score * WEIGHTS['population']
    
    # 3. Internet Score (0-100)
    if row['Internet'] >= internet_percentiles[0.75]:
        internet_score = 100
    elif row['Internet'] >= internet_percentiles[0.5]:
        internet_score = 70
    elif row['Internet'] >= internet_percentiles[0.25]:
        internet_score = 40
    else:
        internet_score = 20
    score += internet_score * WEIGHTS['internet']
    
    # 4. Inflation Penalty (lower inflation = higher score)
    if row['Inflation'] <= 2:
        inflation_score = 100
    elif row['Inflation'] <= 5:
        inflation_score = 80
    elif row['Inflation'] <= 10:
        inflation_score = 50
    elif row['Inflation'] <= 15:
        inflation_score = 30
    else:
        inflation_score = 10
    score += inflation_score * WEIGHTS['inflation']
    
    # 5. Exchange Rate Stability Score
    if pd.notna(row['ExchangeRate']) and row['ExchangeRate'] > 0:
        # Lower variation is better (simplified - using rate magnitude as proxy)
        if row['ExchangeRate'] < 10:
            exchange_score = 100
        elif row['ExchangeRate'] < 50:
            exchange_score = 70
        elif row['ExchangeRate'] < 100:
            exchange_score = 50
        else:
            exchange_score = 30
    else:
        exchange_score = 50
    score += exchange_score * WEIGHTS['exchange']
    
    # 6. Historical Similarity Score
    hist_score = row['Similarity_Best'] * 100
    score += hist_score * WEIGHTS['historical']
    
    return round(score, 1)

# Apply scoring
targets['Market_Score'] = targets.apply(calculate_weighted_score, axis=1)

# Assign priority levels
def get_priority(score):
    if score >= 75:
        return "Very High"
    elif score >= 60:
        return "High"
    elif score >= 45:
        return "Medium"
    elif score >= 30:
        return "Low"
    else:
        return "Very Low"

targets['Priority'] = targets['Market_Score'].apply(get_priority)

# ==========================================
# 7. REALISTIC SALES ESTIMATION WITH PENETRATION
# ==========================================

# Define realistic year-1 penetration rates by priority
PENETRATION_RATES = {
    'Very High': 0.04,  # 4% market capture in year 1
    'High': 0.025,      # 2.5%
    'Medium': 0.012,    # 1.2%
    'Low': 0.005,       # 0.5%
    'Very Low': 0.002    # 0.2%
}

# Economic risk penalties
def calculate_risk_penalty(row):
    """Apply penalties based on economic risks"""
    penalty = 1.0
    
    # Inflation penalty
    if row['Inflation'] > 15:
        penalty *= 0.6
    elif row['Inflation'] > 10:
        penalty *= 0.75
    elif row['Inflation'] > 7:
        penalty *= 0.85
    
    # Exchange rate penalty (high rates = instability)
    if pd.notna(row['ExchangeRate']) and row['ExchangeRate'] > 100:
        penalty *= 0.7
    elif pd.notna(row['ExchangeRate']) and row['ExchangeRate'] > 50:
        penalty *= 0.85
    
    # Low internet penalty
    if row['Internet'] < 30:
        penalty *= 0.6
    elif row['Internet'] < 50:
        penalty *= 0.8
    
    return penalty

def estimate_realistic_sales(row):
    """Estimate year-1 sales with realistic penetration and penalties"""
    # Find similar countries' sales per capita
    similar = historical[
        (historical['GDP'] > row['GDP'] * 0.5) & 
        (historical['GDP'] < row['GDP'] * 1.5) &
        (historical['InternetUsage'] > row['Internet'] * 0.7) &
        (historical['InternetUsage'] < row['Internet'] * 1.3)
    ]
    
    if len(similar) > 0:
        expected_per_capita = similar['SalesPerCapita'].median()
    else:
        expected_per_capita = historical['SalesPerCapita'].median()
    
    # Apply penetration rate
    penetration = PENETRATION_RATES[row['Priority']]
    
    # Apply economic risk penalty
    risk_penalty = calculate_risk_penalty(row)
    
    # Calculate year-1 sales
    year1_sales = expected_per_capita * row['Population'] * penetration * risk_penalty
    
    # Calculate year-3 potential (if risks are managed)
    year3_sales = year1_sales * 2.5 * (1 + (1 - risk_penalty) * 0.5)
    
    return {
        'year1': year1_sales,
        'year3': year3_sales,
        'penetration_rate': penetration * 100,
        'risk_penalty': risk_penalty
    }

print("\n📈 Estimating realistic sales potential...")
sales_estimates = targets.apply(estimate_realistic_sales, axis=1)
targets['Est_Sales_Year1_M'] = [s['year1'] / 1_000_000 for s in sales_estimates]
targets['Est_Sales_Year3_M'] = [s['year3'] / 1_000_000 for s in sales_estimates]
targets['Penetration_Rate'] = [s['penetration_rate'] for s in sales_estimates]
targets['Risk_Penalty'] = [s['risk_penalty'] for s in sales_estimates]

# ==========================================
# 8. EXPANSION DECISION ENGINE
# ==========================================

def make_expansion_decision(row):
    """Make final expansion recommendation with clear reasoning"""
    score = row['Market_Score']
    priority = row['Priority']
    year1_sales = row['Est_Sales_Year1_M']
    risk_penalty = row['Risk_Penalty']
    inflation = row['Inflation']
    internet = row['Internet']
    
    reasons = []
    warnings = []
    
    # Decision logic
    if priority == 'Very High' and risk_penalty > 0.8:
        decision = "EXPAND NOW"
        action_urgency = "Immediate"
        reasoning = f"Exceptional opportunity (score: {score}/100) with strong fundamentals"
        
    elif priority == 'Very High' and risk_penalty <= 0.8:
        decision = "CONSIDER WITH CAUTION"
        action_urgency = "Short-term"
        reasoning = f"Great potential but significant economic risks to mitigate"
        
    elif priority == 'High' and year1_sales > 10 and risk_penalty > 0.7:
        decision = "HIGH PRIORITY"
        action_urgency = "Next Quarter"
        reasoning = f"Strong market opportunity with good risk/reward profile"
        
    elif priority == 'High':
        decision = "PILOT FIRST"
        action_urgency = "Test Entry"
        reasoning = f"Good potential but needs validation or risk mitigation"
        
    elif priority == 'Medium' and year1_sales > 5:
        decision = "PILOT FIRST"
        action_urgency = "Strategic"
        reasoning = f"Moderate opportunity - recommended pilot program"
        
    elif priority == 'Medium':
        decision = "MONITOR"
        action_urgency = "Watch List"
        reasoning = f"Limited near-term opportunity, revisit in 6-12 months"
        
    elif priority == 'Low':
        decision = "WAIT"
        action_urgency = "No Action"
        reasoning = f"Unfavorable market conditions or limited potential"
        
    else:
        decision = "AVOID"
        action_urgency = "None"
        reasoning = f"Poor market fit with significant barriers to success"
    
    # Add specific reasons
    if score >= 70:
        reasons.append(f"Excellent market score ({score}/100)")
    elif score >= 50:
        reasons.append(f"Good market score ({score}/100)")
    
    if year1_sales > 20:
        reasons.append(f"High year-1 potential (${year1_sales:.0f}M)")
    elif year1_sales > 10:
        reasons.append(f"Solid year-1 potential (${year1_sales:.0f}M)")
    
    if row['Population'] > 100_000_000:
        reasons.append(f"Massive addressable market ({row['Population']/1e6:.0f}M people)")
    elif row['Population'] > 50_000_000:
        reasons.append(f"Large market ({row['Population']/1e6:.0f}M people)")
    
    if row['GDP'] > 30000:
        reasons.append(f"High purchasing power (${row['GDP']:,.0f} GDP/capita)")
    
    if row['Internet'] > 70:
        reasons.append(f"Excellent digital infrastructure ({row['Internet']:.0f}% internet)")
    
    # Add warnings
    if risk_penalty < 0.7:
        warnings.append(f"High economic risk (penalty: {(1-risk_penalty)*100:.0f}%)")
    elif risk_penalty < 0.85:
        warnings.append(f"Moderate economic risk")
    
    if inflation > 10:
        warnings.append(f"High inflation ({inflation:.1f}%) affects margins")
    elif inflation > 7:
        warnings.append(f"Elevated inflation ({inflation:.1f}%)")
    
    if row['Internet'] < 40:
        warnings.append(f"Low internet penetration ({row['Internet']:.0f}%) limits digital reach")
    
    # Format final reason
    full_reason = reasoning
    if reasons:
        full_reason += f"\n  ✓ {chr(10) + '  ✓ '.join(reasons[:3])}"
    if warnings:
        full_reason += f"\n  ⚠️ {chr(10) + '  ⚠️ '.join(warnings[:3])}"
    
    return decision, full_reason, action_urgency

# Apply decision engine
decision_results = targets.apply(
    lambda row: make_expansion_decision(row), 
    axis=1, 
    result_type='expand'
)
targets['Expansion_Decision'] = decision_results[0]
targets['Decision_Reason'] = decision_results[1]
targets['Action_Urgency'] = decision_results[2]

# ==========================================
# 9. FINAL RANKINGS
# ==========================================

# Sort and rank
final_results = targets.sort_values(['Market_Score', 'Est_Sales_Year1_M'], ascending=[False, False])
final_results['Rank'] = range(1, len(final_results) + 1)

print("\n" + "="*80)
print("STEP 4: Expansion Recommendations")
print("="*80)

print("\n🎯 TOP 20 MARKETS FOR EXPANSION:")
print("-"*120)
print(f"{'Rank':<4} {'Country':<25} {'Score':<7} {'Priority':<10} {'Decision':<18} {'Year1 ($M)':<12} {'Urgency':<12}")
print("-"*120)

for _, row in final_results.head(20).iterrows():
    print(f"{row['Rank']:<4} {row['Country'][:24]:<25} {row['Market_Score']:<7.0f} "
          f"{row['Priority']:<10} {row['Expansion_Decision']:<18} "
          f"${row['Est_Sales_Year1_M']:>10,.0f}  {row['Action_Urgency']:<12}")

# Detailed reasoning for top 10
print("\n" + "="*80)
print("📋 DETAILED MARKET INTELLIGENCE - TOP 10")
print("="*80)

for _, row in final_results.head(10).iterrows():
    print(f"\n🔹 RANK #{row['Rank']}: {row['Country']}")
    print(f"   Decision: {row['Expansion_Decision']} | Priority: {row['Priority']} | Score: {row['Market_Score']:.0f}/100")
    print(f"   Year 1 Potential: ${row['Est_Sales_Year1_M']:,.0f}M | Year 3: ${row['Est_Sales_Year3_M']:,.0f}M")
    print(f"   Market Size: {row['Population']/1e6:.1f}M people | GDP: ${row['GDP']:,.0f}")
    print(f"   Digital Reach: {row['Internet']:.0f}% internet | Inflation: {row['Inflation']:.1f}%")
    print(f"   Entry Strategy: {row['Action_Urgency']}")
    print(f"\n   📝 Analysis:")
    # Print first 3 lines of reason
    reason_lines = row['Decision_Reason'].split('\n')
    for line in reason_lines[:4]:
        print(f"      {line}")

# ==========================================
# 10. STRATEGIC SUMMARY
# ==========================================

print("\n" + "="*80)
print("STRATEGIC EXPANSION SUMMARY")
print("="*80)

# Group by decision
decision_summary = final_results.groupby('Expansion_Decision').agg({
    'Country': 'count',
    'Est_Sales_Year1_M': 'sum'
}).rename(columns={'Country': 'Count', 'Est_Sales_Year1_M': 'Total_Year1_Sales_M'})

print("\n📊 Decision Breakdown:")
for decision in ['EXPAND NOW', 'HIGH PRIORITY', 'PILOT FIRST', 'CONSIDER WITH CAUTION', 'MONITOR', 'WAIT', 'AVOID']:
    if decision in decision_summary.index:
        row = decision_summary.loc[decision]
        print(f"\n   {decision}:")
        print(f"      → {row['Count']} countries")
        print(f"      → ${row['Total_Year1_Sales_M']:,.0f}M year-1 potential")

# Priority summary
priority_summary = final_results.groupby('Priority').agg({
    'Country': 'count',
    'Est_Sales_Year1_M': 'sum',
    'Est_Sales_Year3_M': 'sum'
}).rename(columns={'Country': 'Count'})

print("\n📊 Priority Level Breakdown:")
for priority in ['Very High', 'High', 'Medium', 'Low', 'Very Low']:
    if priority in priority_summary.index:
        row = priority_summary.loc[priority]
        print(f"\n   {priority.upper()} PRIORITY:")
        print(f"      → {row['Count']} markets")
        print(f"      → ${row['Est_Sales_Year1_M']:,.0f}M year-1")
        print(f"      → ${row['Est_Sales_Year3_M']:,.0f}M year-3 potential")

# ==========================================
# 11. EXPORT RESULTS
# ==========================================

export_columns = [
    'Rank', 'Country', 'Market_Score', 'Priority', 'Expansion_Decision',
    'Action_Urgency', 'Est_Sales_Year1_M', 'Est_Sales_Year3_M',
    'Penetration_Rate', 'Risk_Penalty', 'GDP', 'Population', 'Internet',
    'Inflation', 'ExchangeRate', 'Similarity_Best', 'Decision_Reason'
]

# Ensure all columns exist
available_cols = [col for col in export_columns if col in final_results.columns]
final_results[available_cols].to_csv('market_expansion_intelligence.csv', index=False)
print(f"\n✓ Detailed results exported to 'market_expansion_intelligence.csv'")

# Create executive summary
exec_summary = final_results[final_results['Priority'].isin(['Very High', 'High'])][[
    'Rank', 'Country', 'Priority', 'Expansion_Decision', 'Est_Sales_Year1_M'
]].head(20)
exec_summary.to_csv('executive_summary_top_markets.csv', index=False)
print(f"✓ Executive summary exported to 'executive_summary_top_markets.csv'")

# ==========================================
# 12. VISUALIZATIONS
# ==========================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Create comprehensive dashboard
fig = plt.figure(figsize=(20, 14))

# 1. Priority Distribution
ax1 = fig.add_subplot(2, 3, 1)
priority_counts = final_results['Priority'].value_counts()
priority_colors = {'Very High': '#2ecc71', 'High': '#27ae60', 
                   'Medium': '#f39c12', 'Low': '#e74c3c', 'Very Low': '#c0392b'}
colors = [priority_colors.get(p, '#95a5a6') for p in priority_counts.index]
wedges, texts, autotexts = ax1.pie(priority_counts.values, labels=priority_counts.index,
                                     autopct='%1.0f%%', colors=colors, textprops={'fontsize': 10})
ax1.set_title('Market Priority Distribution', fontsize=14, fontweight='bold')

# 2. Score Distribution
ax2 = fig.add_subplot(2, 3, 2)
ax2.hist(final_results['Market_Score'], bins=20, color='#3498db', edgecolor='black', alpha=0.7)
ax2.axvline(final_results['Market_Score'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f"Mean: {final_results['Market_Score'].mean():.1f}")
ax2.axvline(final_results['Market_Score'].median(), color='orange', linestyle='--', 
            linewidth=2, label=f"Median: {final_results['Market_Score'].median():.1f}")
ax2.set_xlabel('Market Score (0-100)', fontsize=11)
ax2.set_ylabel('Number of Countries', fontsize=11)
ax2.set_title('Distribution of Market Scores', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Top 20 by Score
ax3 = fig.add_subplot(2, 3, 3)
top20 = final_results.head(20)
colors_bar = [priority_colors.get(p, '#95a5a6') for p in top20['Priority']]
bars = ax3.barh(range(len(top20)), top20['Market_Score'], color=colors_bar)
ax3.set_yticks(range(len(top20)))
ax3.set_yticklabels(top20['Country'], fontsize=9)
ax3.set_xlabel('Market Score (0-100)', fontsize=11)
ax3.set_title('Top 20 Markets by Weighted Score', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
# Add value labels
for i, (bar, score) in enumerate(zip(bars, top20['Market_Score'])):
    ax3.text(score + 1, bar.get_y() + bar.get_height()/2, f'{score:.0f}', 
             va='center', fontsize=8)

# 4. Year-1 Sales by Priority
ax4 = fig.add_subplot(2, 3, 4)
priority_sales = final_results.groupby('Priority')['Est_Sales_Year1_M'].sum().sort_values()
sales_colors = [priority_colors.get(p, '#95a5a6') for p in priority_sales.index]
bars = ax4.bar(range(len(priority_sales)), priority_sales.values, color=sales_colors)
ax4.set_xticks(range(len(priority_sales)))
ax4.set_xticklabels(priority_sales.index, rotation=45, ha='right')
ax4.set_ylabel('Total Year-1 Sales (Million USD)', fontsize=11)
ax4.set_title('Year-1 Sales Potential by Priority', fontsize=14, fontweight='bold')
for i, (bar, value) in enumerate(zip(bars, priority_sales.values)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.02,
             f'${value:,.0f}M', ha='center', va='bottom', fontsize=9, rotation=0)

# 5. Decision Distribution
ax5 = fig.add_subplot(2, 3, 5)
decision_counts = final_results['Expansion_Decision'].value_counts()
decision_colors = {'EXPAND NOW': '#2ecc71', 'HIGH PRIORITY': '#27ae60',
                   'PILOT FIRST': '#f39c12', 'CONSIDER WITH CAUTION': '#e67e22',
                   'MONITOR': '#3498db', 'WAIT': '#95a5a6', 'AVOID': '#e74c3c'}
colors_decision = [decision_colors.get(d, '#95a5a6') for d in decision_counts.index]
bars = ax5.bar(range(len(decision_counts)), decision_counts.values, color=colors_decision)
ax5.set_xticks(range(len(decision_counts)))
ax5.set_xticklabels(decision_counts.index, rotation=45, ha='right', fontsize=9)
ax5.set_ylabel('Number of Countries', fontsize=11)
ax5.set_title('Expansion Decisions Distribution', fontsize=14, fontweight='bold')
for i, (bar, count) in enumerate(zip(bars, decision_counts.values)):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom', fontweight='bold')

# 6. Risk vs Reward Matrix
ax6 = fig.add_subplot(2, 3, 6)
# Create risk categories based on penalty
risk_categories = []
for penalty in final_results['Risk_Penalty']:
    if penalty > 0.85:
        risk_categories.append('Low Risk')
    elif penalty > 0.7:
        risk_categories.append('Medium Risk')
    else:
        risk_categories.append('High Risk')
final_results['Risk_Category'] = risk_categories

# Create scatter plot
for risk in ['Low Risk', 'Medium Risk', 'High Risk']:
    subset = final_results[final_results['Risk_Category'] == risk]
    color = {'Low Risk': '#2ecc71', 'Medium Risk': '#f39c12', 'High Risk': '#e74c3c'}[risk]
    size = subset['Est_Sales_Year1_M'] / 10 + 20
    ax6.scatter(subset['Market_Score'], subset['Est_Sales_Year1_M'], 
               s=size, c=color, alpha=0.6, label=risk, edgecolors='black', linewidth=0.5)

ax6.set_xlabel('Market Score (0-100)', fontsize=11)
ax6.set_ylabel('Year-1 Sales Potential (Million USD)', fontsize=11)
ax6.set_title('Risk vs Reward Matrix\n(Bubble size = sales potential)', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)
ax6.set_yscale('log')
ax6.set_ylim(1, max(final_results['Est_Sales_Year1_M']) * 1.1)

plt.suptitle('Market Expansion Intelligence Dashboard', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('market_expansion_dashboard.png', dpi=150, bbox_inches='tight')
print("✓ Saved: market_expansion_dashboard.png")

# Additional chart: Top 15 by Year-1 Sales
fig2, ax = plt.subplots(figsize=(14, 8))
top15_sales = final_results.nlargest(15, 'Est_Sales_Year1_M')
colors_sales = [priority_colors.get(p, '#95a5a6') for p in top15_sales['Priority']]
bars = ax.barh(range(len(top15_sales)), top15_sales['Est_Sales_Year1_M'], color=colors_sales)
ax.set_yticks(range(len(top15_sales)))
ax.set_yticklabels(top15_sales['Country'], fontsize=10)
ax.set_xlabel('Estimated Year-1 Sales (Million USD)', fontsize=12)
ax.set_title('Top 15 Markets by Year-1 Sales Potential', fontsize=14, fontweight='bold')
ax.invert_yaxis()
for i, (bar, value) in enumerate(zip(bars, top15_sales['Est_Sales_Year1_M'])):
    ax.text(bar.get_width() + value*0.02, bar.get_y() + bar.get_height()/2,
            f'${value:,.0f}M', va='center', fontsize=9)
plt.tight_layout()
plt.savefig('top_markets_by_sales_potential.png', dpi=150, bbox_inches='tight')
print("✓ Saved: top_markets_by_sales_potential.png")

# ==========================================
# 13. FINAL EXECUTIVE SUMMARY
# ==========================================

print