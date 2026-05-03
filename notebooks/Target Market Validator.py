import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==========================================
# 1. LOAD DATA
# ==========================================
def load_csv_safe(file_path):
    """Attempts to load a CSV using multiple encodings."""
    for enc in ['utf-8', 'ISO-8859-1', 'latin1', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise Exception(f"Could not decode {file_path} with standard encodings.")

print("="*60)
print("MARKET EXPANSION PRIORITIZATION TOOL")
print("="*60)

print("\nLoading data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
sales_df = load_csv_safe(os.path.join(script_dir, '../data/processed/merged_city_sales_data.csv'))
country_df = load_csv_safe(os.path.join(script_dir, '../data/processed/filtered_data.csv'))

print(f"- Sales data: {sales_df.shape[0]} transactions")
print(f"- Country data: {country_df.shape[0]} countries to evaluate")

# ==========================================
# 2. ANALYZE HISTORICAL PERFORMANCE
# ==========================================
print("\n" + "="*60)
print("STEP 1: Analyzing Historical Market Performance")
print("="*60)

# Aggregate historical sales by country
historical = sales_df.groupby('country_norm_mapped').agg({
    'sales_amount_realistic': 'sum',
    'gdp_per_capita': 'mean',
    'Population': 'mean',
    'internet_usage_pct': 'mean'
}).reset_index()

historical.columns = ['Country', 'TotalSales', 'GDP', 'Population', 'InternetUsage']
historical['SalesPerCapita'] = historical['TotalSales'] / historical['Population']
historical['SalesPerGDP'] = historical['TotalSales'] / historical['GDP']

# Sort by performance
top_performers = historical.sort_values('SalesPerCapita', ascending=False)

print("\n* Top 5 Performing Markets (by Sales Per Capita):")
for i, row in top_performers.head(5).iterrows():
    print(f"   {row['Country']}: ${row['SalesPerCapita']:.2f} per person | GDP: ${row['GDP']:,.0f}")

print(f"\n* Historical Statistics:")
print(f"   Average Sales Per Capita: ${historical['SalesPerCapita'].mean():.2f}")
print(f"   Median Sales Per Capita: ${historical['SalesPerCapita'].median():.2f}")
print(f"   Best Market: {top_performers.iloc[0]['Country']} (${top_performers.iloc[0]['SalesPerCapita']:.2f}/person)")
print(f"   Worst Market: {top_performers.iloc[-1]['Country']} (${top_performers.iloc[-1]['SalesPerCapita']:.2f}/person)")

# ==========================================
# 3. DEFINE SCORING SYSTEM
# ==========================================
print("\n" + "="*60)
print("STEP 2: Scoring System Definition")
print("="*60)

print("""
Scoring Criteria:
┌─────────────────────────────────────────────────────────────┐
│  GDP per capita (Economic strength)                         │
│  • > $50,000  → +3 points (Very High Income)               │
│  • $20k-50k   → +2 points (High Income)                    │
│  • $5k-20k    → +1 point (Middle Income)                   │
│  • < $5,000   → +0 points (Low Income)                     │
├─────────────────────────────────────────────────────────────┤
│  Population (Market size)                                   │
│  • > 100M     → +3 points (Giant market)                   │
│  • 50M-100M   → +2 points (Large market)                   │
│  • 10M-50M    → +1 point (Medium market)                   │
│  • < 10M      → +0 points (Small market)                   │
├─────────────────────────────────────────────────────────────┤
│  Internet Usage (Digital readiness)                         │
│  • > 80%      → +2 points (Highly digital)                 │
│  • 50%-80%    → +1 point (Moderately digital)              │
│  • < 50%      → +0 points (Low digital adoption)           │
├─────────────────────────────────────────────────────────────┤
│  MAXIMUM POSSIBLE SCORE: 8 points                          │
└─────────────────────────────────────────────────────────────┘
""")

# ==========================================
# 4. SCORE NEW MARKETS
# ==========================================
print("\n" + "="*60)
print("STEP 3: Evaluating New Markets")
print("="*60)

# Prepare prediction data
predict_data = country_df.copy()
predict_data.rename(columns={
    'Country Name': 'Country',
    'GDP per capita (USD)': 'GDP',
    'Population': 'Population',
    'Internet Usage (%)': 'Internet',
    'Inflation Rate (%)': 'Inflation',
    'Official Exchange Rate': 'ExchangeRate'
}, inplace=True)

# Ensure Population is numeric
predict_data['Population'] = pd.to_numeric(predict_data['Population'], errors='coerce')

# Define scoring function
def calculate_market_score(row):
    score = 0
    
    # GDP scoring
    if row['GDP'] > 50000:
        score += 3
    elif row['GDP'] > 20000:
        score += 2
    elif row['GDP'] > 5000:
        score += 1
    
    # Population scoring
    if row['Population'] > 100_000_000:
        score += 3
    elif row['Population'] > 50_000_000:
        score += 2
    elif row['Population'] > 10_000_000:
        score += 1
    
    # Internet scoring
    if row['Internet'] > 80:
        score += 2
    elif row['Internet'] > 50:
        score += 1
    
    return score

# Apply scoring
predict_data['Market_Score'] = predict_data.apply(calculate_market_score, axis=1)

# Assign priority tiers
predict_data['Priority'] = pd.cut(predict_data['Market_Score'], 
                                   bins=[-1, 2, 4, 6, 9], 
                                   labels=['Very Low', 'Low', 'Medium', 'High'])

# ==========================================
# 5. ESTIMATE SALES POTENTIAL
# ==========================================
print("\n" + "="*60)
print("STEP 4: Estimating Sales Potential")
print("="*60)

# Use different benchmarks based on market similarity
def estimate_sales(row):
    # Find similar countries from historical data
    similar = historical[
        (historical['GDP'] > row['GDP'] * 0.5) & 
        (historical['GDP'] < row['GDP'] * 2)
    ]
    
    if len(similar) > 0:
        # Use average of similar GDP countries
        expected_per_capita = similar['SalesPerCapita'].median()
    else:
        # Fallback to overall median
        expected_per_capita = historical['SalesPerCapita'].median()
    
    # Adjust for internet penetration
    internet_multiplier = 0.5 + (row['Internet'] / 100)
    
    final_per_capita = expected_per_capita * internet_multiplier
    return final_per_capita * row['Population']

predict_data['Estimated_Annual_Sales'] = predict_data.apply(estimate_sales, axis=1)

# Add confidence level based on GDP similarity
gdp_range = (historical['GDP'].min(), historical['GDP'].max())
predict_data['Confidence'] = np.where(
    (predict_data['GDP'] >= gdp_range[0]) & (predict_data['GDP'] <= gdp_range[1]),
    'High (GDP within known range)',
    'Medium (GDP outside historical range)'
)

# ==========================================
# 6. FINAL RANKINGS
# ==========================================
# Sort by score (primary) and estimated sales (secondary)
final_results = predict_data.sort_values(
    ['Market_Score', 'Estimated_Annual_Sales'], 
    ascending=[False, False]
).reset_index(drop=True)

# Add rank
final_results['Rank'] = final_results.index + 1

# Format currency
final_results['Estimated_Annual_Sales_M'] = final_results['Estimated_Annual_Sales'] / 1_000_000

print("\n🎯 TOP 10 PRIORITY MARKETS FOR EXPANSION:")
print("-"*80)
print(f"{'Rank':<4} {'Country':<25} {'Score':<6} {'Priority':<10} {'Est. Sales (M$)':<15} {'Confidence':<20}")
print("-"*80)

for _, row in final_results.head(10).iterrows():
    print(f"{row['Rank']:<4} {row['Country'][:24]:<25} {row['Market_Score']:<6} {row['Priority']:<10} "
          f"${row['Estimated_Annual_Sales_M']:>12,.0f}  {row['Confidence'][:19]:<20}")

# ==========================================
# 7. RECOMMENDATIONS
# ==========================================
print("\n" + "="*60)
print("STEP 5: Strategic Recommendations")
print("="*60)

high_priority = final_results[final_results['Priority'] == 'High']
medium_priority = final_results[final_results['Priority'] == 'Medium']

print(f"\n📌 Immediate Entry (High Priority): {len(high_priority)} countries")
if len(high_priority) > 0:
    for _, row in high_priority.head(5).iterrows():
        print(f"   → {row['Country']} (Score: {row['Market_Score']}/8 | Est: ${row['Estimated_Annual_Sales_M']:.0f}M)")

print(f"\n📌 Consider Next (Medium Priority): {len(medium_priority)} countries")
if len(medium_priority) > 0:
    for _, row in medium_priority.head(3).iterrows():
        print(f"   → {row['Country']} (Score: {row['Market_Score']}/8 | Est: ${row['Estimated_Annual_Sales_M']:.0f}M)")

# ==========================================
# 8. EXPORT RESULTS
# ==========================================
export_columns = [
    'Rank', 'Country', 'GDP', 'Population', 'Internet', 
    'Market_Score', 'Priority', 'Estimated_Annual_Sales_M', 
    'Confidence', 'Inflation', 'ExchangeRate'
]

final_results[export_columns].to_csv('market_expansion_priorities.csv', index=False)
print(f"\n✓ Full results exported to 'market_expansion_priorities.csv'")

# ==========================================
# 9. VISUALIZATIONS
# ==========================================
print("\n" + "="*60)
print("STEP 6: Generating Visualizations")
print("="*60)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. Priority Distribution (Pie Chart)
ax1 = fig.add_subplot(2, 2, 1)
priority_counts = final_results['Priority'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
wedges, texts, autotexts = ax1.pie(priority_counts.values, labels=priority_counts.index, 
                                     autopct='%1.0f%%', colors=colors[:len(priority_counts)])
ax1.set_title('Market Priority Distribution', fontsize=14, fontweight='bold')

# 2. Top 15 Markets by Score (Horizontal Bar Chart)
ax2 = fig.add_subplot(2, 2, 2)
top15 = final_results.head(15)
colors_bar = ['#2ecc71' if x == 'High' else '#f39c12' if x == 'Medium' else '#e74c3c' 
              for x in top15['Priority']]
bars = ax2.barh(range(len(top15)), top15['Market_Score'], color=colors_bar)
ax2.set_yticks(range(len(top15)))
ax2.set_yticklabels(top15['Country'], fontsize=9)
ax2.set_xlabel('Market Score (0-8)', fontsize=11)
ax2.set_title('Top 15 Markets by Score', fontsize=14, fontweight='bold')
ax2.invert_yaxis()

# Add value labels
for i, (bar, score) in enumerate(zip(bars, top15['Market_Score'])):
    ax2.text(score + 0.1, bar.get_y() + bar.get_height()/2, f'{score}', 
             va='center', fontsize=9)

# 3. Score Components Breakdown (Stacked Bar for Top 10)
ax3 = fig.add_subplot(2, 2, 3)
top10 = final_results.head(10)
gdp_scores = []
pop_scores = []
internet_scores = []

for _, row in top10.iterrows():
    # Recalculate components
    gdp = 3 if row['GDP'] > 50000 else (2 if row['GDP'] > 20000 else (1 if row['GDP'] > 5000 else 0))
    pop = 3 if row['Population'] > 100_000_000 else (2 if row['Population'] > 50_000_000 else (1 if row['Population'] > 10_000_000 else 0))
    internet = 2 if row['Internet'] > 80 else (1 if row['Internet'] > 50 else 0)
    gdp_scores.append(gdp)
    pop_scores.append(pop)
    internet_scores.append(internet)

x = range(len(top10))
ax3.barh(x, gdp_scores, label='GDP', color='#3498db')
ax3.barh(x, pop_scores, left=gdp_scores, label='Population', color='#2ecc71')
ax3.barh(x, internet_scores, left=[gdp_scores[i] + pop_scores[i] for i in range(len(gdp_scores))], 
         label='Internet', color='#e74c3c')
ax3.set_yticks(x)
ax3.set_yticklabels(top10['Country'], fontsize=9)
ax3.set_xlabel('Score Contribution', fontsize=11)
ax3.set_title('What Drives Each Market\'s Score?', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right')
ax3.invert_yaxis()

# 4. Estimated Sales vs Market Score (Bubble Chart)
ax4 = fig.add_subplot(2, 2, 4)
scatter = ax4.scatter(final_results['Market_Score'], 
                      final_results['Estimated_Annual_Sales_M'] / 1000,
                      s=final_results['Population'] / 10_000_000,
                      c=final_results['Market_Score'], 
                      cmap='RdYlGn', 
                      alpha=0.6,
                      edgecolors='black',
                      linewidth=0.5)
ax4.set_xlabel('Market Score', fontsize=11)
ax4.set_ylabel('Estimated Sales (Billions USD)', fontsize=11)
ax4.set_title('Market Score vs Sales Potential\n(Bubble size = Population)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=ax4, label='Score')
ax4.grid(True, alpha=0.3)

plt.suptitle('Market Expansion Prioritization Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('market_priorities_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: market_priorities_analysis.png")

# Additional chart: Top 10 by Estimated Sales
fig2, ax = plt.subplots(figsize=(12, 6))
top10_sales = final_results.head(10)
colors_sales = ['#2ecc71' if x == 'High' else '#f39c12' for x in top10_sales['Priority']]
bars = ax.bar(range(len(top10_sales)), top10_sales['Estimated_Annual_Sales_M'], color=colors_sales)
ax.set_xticks(range(len(top10_sales)))
ax.set_xticklabels(top10_sales['Country'], rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Estimated Annual Sales (Million USD)', fontsize=12)
ax.set_title('Top 10 Markets by Estimated Sales Potential', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, top10_sales['Estimated_Annual_Sales_M'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.02,
            f'${value:.0f}M', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('top_markets_by_sales.png', dpi=150, bbox_inches='tight')
print("✓ Saved: top_markets_by_sales.png")

# ==========================================
# 10. FINAL SUMMARY
# ==========================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  RECOMMENDATION SUMMARY                                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ✅ Immediate Priority (Score 5-8): {len(high_priority)} markets                                 │
│     → Enter these markets first. They have the strongest           │
│       combination of GDP, population, and digital readiness.        │
│                                                                      │
│  ⏰ Secondary Priority (Score 3-4): {len(medium_priority)} markets                                │
│     → Consider after establishing presence in high-priority         │
│       markets. May require localized strategy adjustments.          │
│                                                                      │
│  ⚠️ Low Priority (Score 0-2): {len(final_results[final_results['Priority'] == 'Low'])} markets                                   │
│     → Only enter if strategic reasons exist (e.g., regional hub,    │
│       manufacturing base, or partnership requirements).             │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  NEXT STEPS:                                                        │
│                                                                      │
│  1. Review 'market_expansion_priorities.csv' for complete data      │
│  2. Conduct deeper research on top 5 markets:                       │
│     • Local competition analysis                                    │
│     • Regulatory environment                                        │
│     • Distribution partnerships                                     │
│     • Pricing strategy adaptation                                   │
│  3. Develop market-specific entry plans for high-priority countries│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
""")

print("\n✅ Analysis complete! Generated files:")
print("   📄 market_expansion_priorities.csv - Complete data table")
print("   📊 market_priorities_analysis.png - Multi-chart analysis")
print("   📊 top_markets_by_sales.png - Sales potential chart")