# Revenue Change Analysis: v2 vs v3

## The Question

> Total revenue dropped from **$6.53B (v2)** to **$5.22B (v3)** — a **-20.1% decrease**.  
> Quantity *increased* from 6.26 to 7.76 avg — a **+24.0% increase**.  
> Why did revenue go DOWN when quantity went UP?

---

## The Short Answer

**The revenue drop is almost entirely caused by one thing: Desktop and Laptop categories no longer sell unrealistic volumes.**

In v2, a $6,999 Mac Pro sold the same ~6 units per transaction as a $129 AirPod. In v3, Mac Pros sell ~2 units while AirPods sell ~11. Since Desktops and Laptops have the highest prices, their volume collapse wiped out **$2.3 billion in revenue** — more than enough to explain the $1.3B total drop.

---

## Detailed Decomposition

### Step 1: Revenue = Quantity x Price

| Metric | v2 | v3 | Change |
|---|---|---|---|
| **Avg quantity_realistic** | 6.26 | 7.76 | **+24.0%** |
| **Avg price_realistic** | $977 | $983 | **+0.6%** |
| **Avg transaction revenue** | $6,107 | $4,882 | **-20.1%** |

> [!IMPORTANT]
> Quantity went UP and price stayed the SAME. So how did revenue go DOWN?
> 
> Because the **composition** of what's being sold changed. The average quantity went up because cheap products (Accessories, Audio, Subscriptions) now sell many more units. But the revenue per unit on those cheap products is tiny compared to what was lost from expensive products selling fewer units.

### Step 2: The Category Waterfall

Here's where every dollar of change came from:

| Category | v2 Revenue | v3 Revenue | Change | Qty Change | Why |
|---|---|---|---|---|---|
| **Desktop** | $2,132M | $557M | **-$1,575M** | 6.2 -> 2.0 | Mac Pros ($5999-$6999) now sell realistically low volumes |
| **Laptop** | $1,337M | $647M | **-$690M** | 6.3 -> 3.3 | MacBook Pros ($1999-$2499) sell fewer units realistically |
| **Tablet** | $784M | $550M | **-$234M** | 6.2 -> 4.8 | iPads are mid-price, moderate volume reduction |
| **Streaming** | $22M | $17M | **-$5M** | 6.0 -> 4.7 | Small category, minor impact |
| Smart Speaker | $3M | $3M | ~$0 | 5.9 -> 6.8 | Negligible |
| Subscription | $3M | $11M | **+$8M** | 6.3 -> 24.8 | High volume but very low price ($1-$20) |
| Accessories | $11M | $18M | **+$7M** | 5.9 -> 12.0 | More units but low price ($29-$299) |
| Audio | $92M | $155M | **+$63M** | 6.2 -> 10.7 | AirPods sell well, moderate price |
| Wearable | $281M | $455M | **+$174M** | 6.2 -> 10.4 | Apple Watch is popular |
| **Smartphone** | $1,862M | $2,806M | **+$944M** | 6.3 -> 9.8 | iPhones are mass market, big boost |
| | | | | | |
| **TOTAL** | **$6,528M** | **$5,218M** | **-$1,310M** | | |

**The math is clear:**
- Categories that **lost** revenue: Desktop (-$1,575M) + Laptop (-$690M) + Tablet (-$234M) = **-$2,504M lost**
- Categories that **gained** revenue: Smartphone (+$944M) + Wearable (+$174M) + Audio (+$63M) + others = **+$1,196M gained**
- **Net: -$1,310M** (matches exactly)

---

## Why Each Factor Changed (and Is It Logical?)

### Factor 1: `season_factor`

| | v2 | v3 |
|---|---|---|
| Mean | 1.0382 | 1.0189 |
| Std | 0.1597 | 0.2030 |
| Unique values | 4 | 12 |

**Impact on revenue:** Slight decrease (-1.9% on mean) because v2 had 8 months at exactly 1.0, but v3 has months like Feb (0.80) and Jul (0.87) pulling the average down. This is correct — Apple's real average monthly sales factor is slightly below 1.0 because there are more "slow" months than "peak" months.

**Is it logical?** Yes. Apple genuinely sells less in Feb/Jul/Aug than in Sep/Nov/Dec.

---

### Factor 2: `economic_factor`

| | v2 | v3 |
|---|---|---|
| Mean | 0.9942 | 0.9671 |
| Std | **0.3539** | **0.0878** |

**Impact on revenue:** The mean dropped slightly (-2.7%), but the huge change is in **standard deviation** (0.35 -> 0.09). 

In v2, some rows had `economic_factor = 1.5` (the maximum) while others had `0.5` (the minimum). This extreme spread was caused by dividing GDP by a global mean that mixed rich and poor countries. A US store with $64K GDP divided by a global mean of ~$40K gave factor ~1.6, while Mexico with $8.8K gave ~0.22 (clipped to 0.5).

In v3, each country is normalized against *itself*, so the factor stays much closer to 1.0 for everyone.

**Is it logical?** Yes. A Thai Apple Store doesn't sell 3x fewer units than a US store just because Thailand's GDP is lower — Apple's prices are already adjusted for local markets. Per-country normalization is correct.

---

### Factor 3: `promo_factor`

| | v2 | v3 |
|---|---|---|
| Mean | 1.0400 | 1.0514 |
| Promo rate | 10.0% | 15.8% |

**Impact on revenue:** Slightly higher (+1.1%). More promotions (15.8% vs 10%) with variable discount factors (1.15-1.50 instead of fixed 1.40). The increase in promo rate slightly boosts demand.

**Is it logical?** Yes. Real Apple retail has ~15-20% of transactions on some form of promotion (education pricing, trade-in deals, holiday bundles).

---

### Factor 4: `trend_factor`

| | v2 | v3 |
|---|---|---|
| Mean | 1.1404 | 1.1600 |
| Std | **0.3617** | **0.2385** |

**Impact on revenue:** Mean is similar (+1.7%), but v3 has less extreme variance. The old trend factors were random per-product with no category logic, creating wild swings. New ones are anchored to realistic category growth rates.

**Is it logical?** Yes. Wearables and subscriptions genuinely trend upward, while streaming devices and smart speakers are declining.

---

### Factor 5: Category-Aware Demand Scaling (NEW in v3)

This is the **biggest driver** of the revenue change. v3 introduces two new multipliers in `mu_demand`:

```
cat_base:    Category scaling (1.4 for phones, 0.5 for desktops)
price_scale: Price-inverse scaling (2.0 for <$100, 0.4 for >$2000)
```

These didn't exist in v2. In v2, `mu_demand = quantity * season * economic * promo * trend * store`, treating all categories equally. In v3, a $29 AirTag has a `cat_base * price_scale` of `1.2 * 2.0 = 2.4`, while a $6,999 Mac Pro has `0.5 * 0.4 = 0.2`. That's a **12x difference** in base demand between them.

**Is it logical?** Absolutely. Apple sells millions of AirPods per quarter and only thousands of Mac Pros. This is the single most important fix for realism.

---

### Factor 6: Shock Events

| Event | Period | Multiplier | Revenue Impact |
|---|---|---|---|
| COVID aftermath | 2021 Q1 | 0.85 | -15% on 53K rows |
| **Supply chain crisis** | **2022 Feb-Jun** | **0.75** | **-48% avg revenue in that window** |
| Tech layoffs pullback | 2023 Q1 | 0.90 | -10% on 53K rows |
| iPhone 16 boost | 2024 Q4 smartphones | 1.15 | +15% on 23K rows |

The 2022 supply chain shock is the most visible: revenue in Feb-Jun 2022 dropped from $5,067/transaction (v2) to $2,657/transaction (v3), a **-48% decrease** in that window. This matches real-world data where Apple reported supply constraints and revenue misses in that period.

**Is it logical?** Yes. Apple's Q2-Q3 2022 earnings reports specifically cited supply chain disruptions costing billions in lost sales.

---

## Yearly Revenue Trend

| Year | v2 | v3 | Change | Why |
|---|---|---|---|---|
| 2021 | $1.12B | $0.72B | -36% | COVID aftermath shock + earliest trend factors |
| 2022 | $1.14B | $0.80B | -30% | Supply chain crisis (Feb-Jun) hit hardest |
| 2023 | $1.43B | $1.04B | -27% | Recovery but still below v2's inflated baseline |
| 2024 | $1.33B | $1.19B | -10% | Gap narrows, iPhone 16 boost helps |
| 2025 | $1.51B | $1.47B | -3% | Nearly equal — trend factors have caught up |

> [!NOTE]
> The gap between v2 and v3 **narrows over time** (from -36% in 2021 to -3% in 2025). This is because:
> 1. Trend factors accumulate over time, boosting demand in later years
> 2. Newer products (2024-2025) are mostly smartphones and wearables, which have HIGH category scaling
> 3. Shock events only hit 2021-2023
> 
> This is a sign the model is working correctly — not that something is broken.

---

## Why Quantity Went UP But Revenue Went DOWN

Think of it like a grocery store analogy:

| | v2 (Before) | v3 (After) |
|---|---|---|
| **Scenario** | Every customer buys 6 of everything (6 bananas, 6 steaks, 6 TVs) | Customers buy 25 bananas, 10 steaks, 2 TVs |
| **Total items** | 18 items | 37 items (**more**) |
| **Total revenue** | 6x$0.50 + 6x$15 + 6x$500 = **$3,093** | 25x$0.50 + 10x$15 + 2x$500 = **$1,163** |

More items were sold, but the expensive items (TVs/Desktops/Laptops) dropped from 6 to 2, crushing the total revenue.

In our data:
- **Subscriptions** went from 6.3 -> 24.8 units (+293%), but at $1-$20 each, that only adds $8M
- **Desktops** went from 6.2 -> 2.0 units (-68%), and at $600-$6,999 each, that removes $1,575M

The net effect: **+24% more units, but -20% less revenue**.

---

## Final Verdict: Is v3 Correct?

| Question | Answer |
|---|---|
| Is the revenue drop logical? | **Yes** — caused by realistic category demand scaling |
| Are the quantities realistic? | **Yes** — cheap products sell more, expensive sell less |
| Does seasonality look right? | **Yes** — Sep/Nov/Dec peaks, Feb dip |
| Are shock events visible? | **Yes** — 2022 Q1-Q2 is clearly visible |
| Do prices make sense? | **Yes** — mean price is nearly identical ($977 vs $983) |
| Does revenue grow over time? | **Yes** — $0.72B (2021) to $1.47B (2025) |
| Should we revert any changes? | **No** — v3 is objectively more realistic |

> [!TIP]
> If the total revenue feels too low for a global Apple operation, remember that this dataset covers only **75 retail stores** across **19 countries**. Apple's actual annual revenue is ~$400B across all channels (online, carrier, wholesale). $1B/year from 75 stores is roughly **$13.3M per store per year**, which is actually in the right ballpark for Apple Store revenue.
