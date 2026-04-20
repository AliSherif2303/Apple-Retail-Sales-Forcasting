"""
Script to fix the plot_forecast function in COMBINED_ADABOOST_CATBOOST.ipynb
so that actual numbers (like 300,000) appear at the end of each bar
instead of scientific notation.
"""
import json
import os

notebook_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "notebooks",
    "COMBINED_ADABOOST_CATBOOST.ipynb"
)

# Read the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The new plot_forecast function source lines (as a single string)
new_plot_forecast_code = '''\
def plot_forecast(forecast_df, model_name):
    """Visualise the 12-month forecast."""
    monthly_total = forecast_df.groupby('date')['predicted_sales'].sum().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle(f'{model_name} \\u2014 12-Month Forecast (Jan\\u2013Dec 2025)',
                 fontsize=16, fontweight='bold')

    # Left: monthly totals
    ax = axes[0]
    bars = ax.bar(monthly_total['date'].dt.strftime('%Y-%m'), monthly_total['predicted_sales'],
                  color=sns.color_palette('viridis', 12))
    ax.set_title('Total Predicted Sales by Month')
    ax.set_ylabel('Total Sales')
    ax.tick_params(axis='x', rotation=45)
    ax.ticklabel_format(style='plain', axis='y')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:,.0f}',
                ha='center', va='bottom', fontsize=8, rotation=45)

    # Right: top 10 stores total
    top_stores = (forecast_df.groupby('store_name')['predicted_sales']
                  .sum().nlargest(10).sort_values())
    ax = axes[1]
    top_stores.plot.barh(ax=ax, color=sns.color_palette('magma', 10))
    ax.set_title('Top 10 Stores \\u2014 Total Predicted Sales (12 months)')
    ax.set_xlabel('Predicted Sales')
    ax.ticklabel_format(style='plain', axis='x')
    for i, (val, name) in enumerate(zip(top_stores.values, top_stores.index)):
        ax.text(val, i, f'  {val:,.0f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Show sample table
    display(forecast_df.head(12))'''

# Find and replace the cell containing plot_forecast
found = False
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = cell["source"]
    # Handle both list-of-strings and single-string formats
    if isinstance(source, list):
        joined = "".join(source)
    else:
        joined = source
    
    if "def plot_forecast(" in joined:
        # Split into lines for processing
        all_lines = joined.split("\n")

        # Find where plot_forecast starts
        start_idx = None
        for i, line in enumerate(all_lines):
            if line.strip().startswith("def plot_forecast("):
                start_idx = i
                break

        if start_idx is None:
            print("ERROR: Could not find plot_forecast function definition")
            break

        # Keep everything before plot_forecast
        before_lines = all_lines[:start_idx]
        before = "\n".join(before_lines)
        if before and not before.endswith("\n"):
            before += "\n"

        # Build the new source
        new_source = before + new_plot_forecast_code
        cell["source"] = [new_source]
        found = True
        print("OK: Successfully updated plot_forecast function!")
        break

if not found:
    print("ERROR: Could not find the plot_forecast cell")
else:
    # Write back
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"OK: Notebook saved: {notebook_path}")
