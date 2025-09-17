# Customer Churn Analysis Report
_Generated on 2025-09-17 22:49 UTC_

## Executive Brief
**Snapshot**
- Overall churn rate sits at 38.0% (~1 in 3 customers) with monthly revenue of $9,452.
- 46 customers exceed the 60% risk threshold, representing $20,231 in probability-weighted six-month exposure.
**Top churn drivers**
- last_login_days: correlation 0.20, mean delta 6.98
- tenure_months: correlation -0.18, mean delta -3.78
- monthly_spend: correlation -0.10, mean delta -1.88
- Basic plan members churn most often at 47.8%, signalling pricing or experience gaps.
**Next steps**
1. Target inactive, short-tenure, low-spend customers with a re-engagement campaign and personalised incentives before they cancel.
2. Prioritise Basic-plan outreach and review packaging to reduce the 48% churn rate.
3. Bolster support experience for heavy-ticket users (3-5 tickets) who churn at 47%.
4. Track KPI impact using the rule-based alerts (69% recall) and recalibrate the logistic model for precision targeting.

## 1. Business Snapshot
- Total customers analyzed: 500
- Overall churn rate: 38.0% (approximately 1 in 3 customers)
- Monthly recurring revenue: $9,452
- High-risk customers (probability = 60%): 46 (9.2% of base)
- Probability-weighted revenue at risk over 6 months: $20,231

### Cohort Averages (0 = retained, 1 = churned)
```
         tenure_months  monthly_spend  support_tickets  last_login_days
churned                                                                
0                20.28          19.62             1.40            27.88
1                16.50          17.74             1.63            34.86
```

## 2. Driver Analysis
- Feature ranking is based on Pearson correlation with churn, mean differences vs. retained customers, and mutual information.
```
                 correlation  mean_delta_vs_retained  mutual_information
last_login_days         0.20                    6.98                0.03
tenure_months          -0.18                   -3.78                0.01
monthly_spend          -0.10                   -1.88                0.02
support_tickets         0.09                    0.23                0.00
```

### 2.1 Correlation Matrix
```
                 tenure_months  monthly_spend  support_tickets  last_login_days  churned
tenure_months             1.00          -0.04             0.03             0.02    -0.18
monthly_spend            -0.04           1.00            -0.06             0.00    -0.10
support_tickets           0.03          -0.06             1.00             0.04     0.09
last_login_days           0.02           0.00             0.04             1.00     0.20
churned                  -0.18          -0.10             0.09             0.20     1.00
```

### 2.2 Mutual Information with Churn
```
                 mutual_information
last_login_days                0.03
monthly_spend                  0.02
tenure_months                  0.01
support_tickets                0.00
```

### 2.3 Logistic Regression Coefficients (odds ratios)
```
           feature  coefficient  odds_ratio
5    plan=Standard        -0.46        0.63
3  last_login_days         0.43        1.54
0    tenure_months        -0.41        0.67
1    monthly_spend        -0.21        0.81
2  support_tickets         0.18        1.20
4     plan=Premium         0.05        1.05
```

## 3. Segment Spotlight
### 3.1 Tenure bucket churn profile
```
               churn_rate  avg_monthly_spend  avg_support_tickets  avg_churn_probability  customers
tenure_bucket                                                                                      
0-6                  0.52              20.52                 1.40                   0.49         73
7-12                 0.39              20.05                 1.31                   0.48         72
13-24                0.43              17.64                 1.59                   0.39        157
25-36                0.29              18.90                 1.51                   0.30        198
36+                   NaN                NaN                  NaN                    NaN          0
```

### 3.2 Inactivity (days since last login)
```
                   churn_rate  avg_monthly_spend  avg_support_tickets  avg_churn_probability  customers
inactivity_bucket                                                                                      
0-14 days                0.25              17.77                 1.41                   0.26        112
15-30                    0.32              20.26                 1.41                   0.32        127
31-60                    0.47              18.73                 1.56                   0.46        261
60+                       NaN                NaN                  NaN                    NaN          0
```

### 3.3 Support ticket volume
```
                churn_rate  avg_monthly_spend  avg_last_login_days  avg_churn_probability  customers
support_bucket                                                                                      
0                     0.38              20.04                28.93                   0.34        116
1-2                   0.35              18.57                31.60                   0.38        284
3-5                   0.47              18.88                29.49                   0.42         96
6+                    0.50              10.51                25.75                   0.60          4
```

### 3.4 Plan profile summary
```
           churn_rate  avg_monthly_spend  avg_support_tickets  avg_churn_probability  customers
plan_type                                                                                      
Basic            0.48              10.11                 1.56                   0.48        186
Premium          0.37              33.13                 1.37                   0.37         94
Standard         0.30              20.26                 1.47                   0.30        220
```

## 4. Prediction Approaches
### Rule-Based Heuristic
Flag a customer if at least two of the following conditions are true:
  - tenure_months <= 16.0
  - monthly_spend <= $15.96
  - support_tickets >= 2
  - last_login_days >= 39

Evaluation (entire dataset):
- Accuracy: 60.6%
- Precision: 48.7%
- Recall: 69.5%
- Confusion matrix counts (TP/TN/FP/FN): 132/171/139/58

### Logistic Regression Benchmark
- Trained on a 70/30 train-test split with scaled numeric features and one-hot encoded plan type.
- Accuracy: 62.0%
- Precision: 50.0%
- Recall: 35.1%
- F1 score: 41.2%
- Confusion matrix counts (TP/TN/FP/FN): 20/73/20/37

Classification report (test set):
```
precision    recall  f1-score   support

           0       0.66      0.78      0.72        93
           1       0.50      0.35      0.41        57

    accuracy                           0.62       150
   macro avg       0.58      0.57      0.57       150
weighted avg       0.60      0.62      0.60       150
```

## 5. Revenue Impact & High-Risk Cohort
- Customers flagged as high risk (? 60% probability): 46 averaging 67% churn likelihood.
- Revenue exposure if high-risk customers churn (6-month projection): $3,285.
- Probability-weighted loss estimate (6-month projection): $20,231.

### Highest-Risk Customers (Top 10 by Logistic Probability)
```
     customer_id  logistic_probability  rule_flag  tenure_months  monthly_spend  support_tickets  last_login_days plan_type
328          329                  0.82          1              9           9.35                5               59     Basic
496          497                  0.79          1             16           5.94                6               54     Basic
467          468                  0.77          1              8           9.91                4               53     Basic
222          223                  0.74          1              2          10.51                2               50     Basic
268          269                  0.74          1              3          13.72                3               48     Basic
146          147                  0.73          1              3          13.69                2               52     Basic
12            13                  0.73          1              2           9.96                4               34     Basic
314          315                  0.72          1              6           8.69                1               55     Basic
188          189                  0.71          1             11          14.61                6               38     Basic
277          278                  0.71          1              2          14.85                1               53     Basic
```

## 6. Synthetic Data Validation
- Generate stress-test cohorts with `python generate_synthetic_churn.py --rows 1000 --output synthetic_customer_churn.csv`.
- Temporarily replace `customer_churn_data.csv` with the synthetic file and rerun `python Existing.py` to compare churn rate, driver ranking, and model performance.
- Watch for deviations (e.g., higher synthetic churn or different leading drivers) to understand how tactics should adjust.

## 7. Visual Assets
- PNG figures are exported to `reports/figures/` for slide decks or the dashboard; inline images are omitted here to keep the markdown portable.

## 8. Recommendations & Next Steps
1. Prioritize outreach for short-tenure, low-spend customers who recently filed multiple tickets and have been inactive; they trigger most rule conditions.
2. Launch a re-engagement sequence for customers inactive 30+ days, combining personalized offers with content recommendations.
3. Use the logistic regression scores to focus retention efforts on the top third of customers by churn probability and A/B test targeted offers.
4. Collect additional behavioral signals (content engagement, satisfaction scores) and validate the models on a future cohort or synthetic holdout set before deployment.

## 9. Deliverables
- Executive summary: `churn_analysis_report.md`
- HTML report: `churn_analysis_report.html`
- Visualizations directory: `figures/`
- Customer-level scores: `churn_risk_scoring.csv`
- Metrics summary (Streamlit ready): `metrics_summary.json`
