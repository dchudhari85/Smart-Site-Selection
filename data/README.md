{
  "about": "Synthetic dataset for Clinical Trial Site Selection incl. PIs and site performance",
  "version": "v1",
  "new_trial": {
    "therapeutic_area": "Oncology",
    "indication": "NSCLC",
    "phase": "III"
  },
  "files": [
    "sites.csv",
    "principal_investigators.csv",
    "site_performance_history.csv",
    "feasibility_responses_new_trial.csv",
    "recommended_top_sites.csv"
  ],
  "scoring_weights": {
    "avg_enroll_rate_per_month_scaled": 0.35,
    "screen_fail_rate_scaled": -0.15,
    "protocol_deviation_rate_scaled": -0.15,
    "data_entry_lag_days_scaled": -0.1,
    "retention_rate_scaled": 0.25,
    "competing_trials_same_ta_scaled": -0.1
  }
}