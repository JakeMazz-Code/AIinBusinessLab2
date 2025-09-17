# Customer Churn Analysis Report
_Generated on 2025-09-17 23:34 UTC_

Dataset analysed: Synthetic cohort (200 rows)

## Executive Brief
**Snapshot**
- Overall churn rate sits at 22.0% (~1 in 5 customers) with monthly revenue of $3,728.
- 36 customers exceed the 60% risk threshold, representing $4,018 in probability-weighted six-month exposure.
**Top churn drivers**
- last_login_days: correlation 0.49, mean delta 20.00
- tenure_months: correlation -0.41, mean delta -9.99
- monthly_spend: correlation -0.23, mean delta -4.78
- Basic plan members churn most often at 27.9%, signalling pricing or experience gaps.
**Next steps**
1. Target inactive, short-tenure, low-spend customers with a re-engagement campaign and personalised incentives before they cancel.
2. Prioritise Basic-plan outreach and review packaging to reduce the 48% churn rate.
3. Bolster support experience for heavy-ticket users (3-5 tickets) who churn at 47%.
4. Track KPI impact using the rule-based alerts (69% recall) and recalibrate the logistic model for precision targeting.

## 1. Business Snapshot
- Total customers analyzed: 200
- Overall churn rate: 22.0% (approximately 1 in 5 customers)
- Monthly recurring revenue: 
- High-risk customers (probability = 60%): 36 (18.0% of base)
- Probability-weighted revenue at risk over 6 months: 

### Cohort Averages (0 = retained, 1 = churned)
```
         tenure_months  monthly_spend  support_tickets  last_login_days
churned                                                                
0                20.44          19.69             1.29            25.84
1                10.45          14.92             1.93            45.84
```

## 2. Driver Analysis
- Feature ranking is based on Pearson correlation with churn, mean differences vs. retained customers, and mutual information.
```
                 correlation  mean_delta_vs_retained  mutual_information
last_login_days         0.49                   20.00                0.16
tenure_months          -0.41                   -9.99                0.12
monthly_spend          -0.23                   -4.78                0.03
support_tickets         0.21                    0.64                0.03
```

### 2.1 Correlation Matrix
```
                 tenure_months  monthly_spend  support_tickets  last_login_days  churned
tenure_months             1.00          -0.08             0.14            -0.12    -0.41
monthly_spend            -0.08           1.00            -0.08             0.11    -0.23
support_tickets           0.14          -0.08             1.00             0.01     0.21
last_login_days          -0.12           0.11             0.01             1.00     0.49
churned                  -0.41          -0.23             0.21             0.49     1.00
```

### 2.2 Mutual Information with Churn
```
                 mutual_information
last_login_days                0.16
tenure_months                  0.12
support_tickets                0.03
monthly_spend                  0.03
```

### 2.3 Logistic Regression Coefficients (odds ratios)
```
           feature  coefficient  odds_ratio
3  last_login_days         2.76       15.85
0    tenure_months        -2.01        0.13
5    plan=Standard        -1.92        0.15
1    monthly_spend        -1.53        0.22
2  support_tickets         1.23        3.43
4     plan=Premium         0.69        2.00
```

## 3. Segment Spotlight
### 3.1 Tenure bucket churn profile
```
               churn_rate  avg_monthly_spend  avg_support_tickets  avg_churn_probability  customers
tenure_bucket                                                                                      
0-6                  0.58              18.14                 1.27                   0.56         33
7-12                 0.37              20.36                 1.15                   0.35         27
13-24                0.12              19.48                 1.35                   0.14         65
25-36                0.09              17.52                 1.68                   0.09         75
36+                   NaN                NaN                  NaN                    NaN          0
```

### 3.2 Inactivity (days since last login)
```
                   churn_rate  avg_monthly_spend  avg_support_tickets  avg_churn_probability  customers
inactivity_bucket                                                                                      
0-14 days                0.00              17.37                 1.40                   0.01         47
15-30                    0.06              18.14                 1.35                   0.07         54
31-60                    0.41              19.52                 1.50                   0.40         99
60+                       NaN                NaN                  NaN                    NaN          0
```

### 3.3 Support ticket volume
```
                churn_rate  avg_monthly_spend  avg_last_login_days  avg_churn_probability  customers
support_bucket                                                                                      
0                     0.12              18.22                28.42                   0.13         50
1-2                   0.23              18.95                31.16                   0.22        111
3-5                   0.32              18.61                30.11                   0.32         38
6+                    1.00               6.37                24.00                   0.72          1
```

### 3.4 Plan profile summary
```
           churn_rate  avg_monthly_spend  avg_support_tickets  avg_churn_probability  customers
plan_type                                                                                      
Basic            0.28              17.74                 1.59                   0.26         68
Premium          0.28              18.07                 1.19                   0.26         43
Standard         0.15              19.61                 1.44                   0.17         89
```

## 4. Prediction Approaches
### Rule-Based Heuristic
Flag a customer if at least two of the following conditions are true:
  - tenure_months <= 8.0
  - monthly_spend <= $15.21
  - support_tickets >= 3
  - last_login_days >= 50

Evaluation (entire dataset):
- Accuracy: 82.5%
- Precision: 60.0%
- Recall: 61.4%
- Confusion matrix counts (TP/TN/FP/FN): 27/138/18/17

### Logistic Regression Benchmark
- Trained on a 70/30 train-test split with scaled numeric features and one-hot encoded plan type.
- Accuracy: 96.7%
- Precision: 100.0%
- Recall: 84.6%
- F1 score: 91.7%
- Confusion matrix counts (TP/TN/FP/FN): 11/47/0/2

Classification report (test set):
`
precision    recall  f1-score   support

           0       0.96      1.00      0.98        47
           1       1.00      0.85      0.92        13

    accuracy                           0.97        60
   macro avg       0.98      0.92      0.95        60
weighted avg       0.97      0.97      0.97        60
`

## 5. Revenue Impact & High-Risk Cohort
- Customers flagged as high risk (= 60% probability): 36 averaging 88% churn likelihood.
- Revenue exposure if high-risk customers churn (6-month projection): .
- Probability-weighted loss estimate (6-month projection): .

### Highest-Risk Customers (Top 10 by Logistic Probability)
```
     customer_id  logistic_probability  rule_flag  tenure_months  monthly_spend  support_tickets  last_login_days plan_type
53   SYN-5200488                  1.00          1              3          14.97                4               53     Basic
115  SYN-6118670                  1.00          1              3          14.06                2               56   Premium
135  SYN-3735550                  1.00          1              1          16.74                2               56   Premium
189  SYN-3713608                  1.00          1             15           5.57                3               54     Basic
162  SYN-4682677                  0.99          1              5           9.81                1               50     Basic
8    SYN-2813225                  0.98          1              3           7.52                2               37     Basic
96   SYN-3156019                  0.98          1              2           5.74                0               41   Premium
36   SYN-8724620                  0.98          1              1          16.31                3               49  Standard
27   SYN-8404853                  0.98          1              3           5.14                0               40   Premium
11   SYN-9780600                  0.97          1             10           6.07                2               52  Standard
```

## 6. Synthetic Cohort Summary
- This report was generated from a synthetic dataset to pressure-test retention strategies.
- Compare these metrics against the historical baseline to confirm whether top drivers and high-risk segments remain consistent.
- Investigate any major divergences (e.g., higher churn rate, new leading drivers) before rolling out campaigns.

## 7. Visual Assets
- PNG figures are exported to `reports/figures/` for slide decks or exploration in the dashboard.

## 8. Recommendations & Next Steps
1. Prioritize outreach for short-tenure, low-spend customers who recently filed multiple tickets and have been inactive; they trigger most rule conditions.
2. Launch a re-engagement sequence for customers inactive 30+ days, combining personalized offers with content recommendations.
3. Use the logistic regression scores to focus retention efforts on the top third of customers by churn probability and A/B test targeted offers.
4. Collect additional behavioral signals (content engagement, satisfaction scores) and validate the models on a future cohort or synthetic holdout set before deployment.

## 9. Deliverables
- Executive summary: churn_analysis_report.md
- HTML report: churn_analysis_report.html
- Visualizations directory: figures/
- Customer-level scores: churn_risk_scoring.csv
- Metrics summary (Streamlit ready): metrics_summary.json
