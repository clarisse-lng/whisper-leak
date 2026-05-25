#!/usr/bin/env python3
"""
Generates prompts/conversations/conversations.json for the conversational Whisper Leak dataset.

Output structure:
  - positive:        100 Python Data Science sessions  (repeat=100, ~10,000 total captures)
  - negative_general: 9,000 Quora-based sessions       (repeat=1)
  - negative_code:   1,000 adversarial code sessions   (repeat=1, 200 per theme × 5 themes)

Each session has 3-5 turns. Turn 1 is the root task (fixed). Turns 2-N are sampled from
a per-session follow-up pool and randomly shuffled to eliminate positional patterns.

Usage:
    python generate_conversation_prompts.py [--output PATH] [--seed N] [--neg-general-count N]
"""

import argparse
import json
import os
import random
import sys

# ── Positive: 100 Python Data Science root tasks with per-task follow-up pools ─────────────────

POSITIVE_SESSIONS_DATA = [
    # ── Data Loading & Cleaning ──────────────────────────────────────────────────────────────
    {
        "root": "Write a pandas function to load a CSV file and impute missing numeric values with column medians and missing categorical values with the mode.",
        "followups": [
            "Optimize this to process files larger than available RAM using chunked reading with configurable chunk size.",
            "I'm getting a ValueError: could not convert string to float on the age column. Debug the issue and write pytest unit tests.",
            "Refactor into a scikit-learn compatible transformer class with fit and transform methods.",
            "Extend it to also export a completeness report showing per-column null rates before and after imputation.",
        ],
    },
    {
        "root": "Create a pandas pipeline to merge two DataFrames on multiple keys using a left join and resolve duplicate column names automatically.",
        "followups": [
            "The merge is producing unexpected duplicates. Add diagnostic logging to trace the source and fix the issue.",
            "Optimize this for two DataFrames of 5M rows each — current runtime is too slow.",
            "Write unit tests covering: empty left DataFrame, keys not present in right, and duplicate key values.",
            "Extend to support outer join with a summary of unmatched rows from both sides.",
        ],
    },
    {
        "root": "Build a function to detect and remove duplicate rows in a DataFrame based on a configurable subset of business-key columns, keeping the most recent entry by a timestamp column.",
        "followups": [
            "Extend this to instead of dropping duplicates, flag them in a new 'is_duplicate' column and produce a report.",
            "I'm getting a KeyError when the timestamp column is named differently. Make the column names configurable with sensible defaults.",
            "Write pytest unit tests covering: no duplicates, all duplicates, missing timestamp column.",
            "Optimize for a 20M row DataFrame — the current groupby is too slow.",
        ],
    },
    {
        "root": "Write a function to parse date strings in multiple formats (ISO 8601, US MM/DD/YYYY, European DD/MM/YYYY) within a DataFrame column into a unified datetime dtype.",
        "followups": [
            "Some dates are ambiguous between US and European format. Add a parameter to specify which to prefer and log ambiguous cases.",
            "I'm getting NaT for dates formatted as '15 January 2023'. Fix the parser to handle written-out month names.",
            "Write unit tests for: ISO format, US format, European format, ambiguous date, completely unparseable string.",
            "Extend to also extract year, month, day, day_of_week, and is_weekend into separate columns.",
        ],
    },
    {
        "root": "Create a function to reshape a wide-format survey DataFrame into long format using pandas melt, preserving all identifier columns and producing clean variable and value column names.",
        "followups": [
            "The melted DataFrame has 10M rows and is slow to filter. Add an option to filter to specific variable names before melting.",
            "I'm getting duplicate identifier combinations after the melt. Diagnose the cause and fix it.",
            "Write unit tests for: single id column, multiple id columns, no value columns to melt.",
            "Extend to also pivot back from long to wide format with the same clean naming conventions.",
        ],
    },
    {
        "root": "Write a function to clean a text column by stripping whitespace, lowercasing, removing special characters, and collapsing multiple spaces into one.",
        "followups": [
            "The cleaning is removing accented characters like é and ñ that should be preserved. Fix it to handle Unicode properly.",
            "Add an option to apply stemming or lemmatization using NLTK after the basic cleaning.",
            "Write unit tests covering: empty string, all special characters, mixed Unicode, already clean text.",
            "Extend to also detect and flag rows where the cleaned text is empty or below a minimum length threshold.",
        ],
    },
    {
        "root": "Implement a function to split a DataFrame column containing pipe-delimited multi-value strings into multiple rows, one value per row, preserving all other columns.",
        "followups": [
            "Some cells contain extra whitespace around the delimiter. Fix the splitting and strip each resulting value.",
            "After exploding, I need to rejoin back to the original format for a specific subset of rows. Add a collapse function.",
            "Write unit tests for: single value, multiple values, empty string, NaN cell.",
            "The explode produces a very large DataFrame. Add a filter parameter to only explode rows matching a condition.",
        ],
    },
    {
        "root": "Create a function to detect and cap outliers in numeric columns using the IQR method with configurable whisker multipliers, returning both the capped DataFrame and a summary of changes.",
        "followups": [
            "Add an alternative z-score based method and let the caller choose between IQR and z-score at runtime.",
            "I'm getting unexpected results when a column has near-zero variance. Handle this edge case gracefully.",
            "Write unit tests for: no outliers, all outliers, single-value column, NaN-heavy column.",
            "Extend to also produce a visualization of before/after distributions for each affected column.",
        ],
    },
    {
        "root": "Write a function to read multiple CSV files from a directory matching a glob pattern, concatenate them into a single DataFrame, and add a source_file column.",
        "followups": [
            "Some files have slightly different column sets. Add a parameter to control whether to use union or intersection of columns.",
            "I'm getting a memory error when concatenating 500 files. Rewrite to use an iterator that yields partial DataFrames.",
            "Write unit tests for: empty directory, single file, files with mismatched columns.",
            "Add parallel reading using concurrent.futures to speed up loading of many large files.",
        ],
    },
    {
        "root": "Build a function to normalize DataFrame column names: convert to snake_case, remove accented characters, strip leading/trailing underscores, and ensure uniqueness.",
        "followups": [
            "Two columns end up with the same normalized name. Add a deduplication suffix strategy and log the collisions.",
            "The function crashes on columns that are integers rather than strings. Handle non-string column names gracefully.",
            "Write unit tests for: already normalized names, mixed case with spaces, accented characters, duplicate normalized names.",
            "Add a mapping output so the caller can trace which original column name became which normalized name.",
        ],
    },
    {
        "root": "Implement a function to validate a DataFrame against a schema dict specifying expected dtypes, non-null constraints, value ranges, and allowed categories, returning a structured validation report.",
        "followups": [
            "The report is hard to read as a dict. Format it as a pandas DataFrame with one row per validation failure.",
            "Add support for cross-column constraints, e.g., column A must be less than column B.",
            "Write unit tests for: fully valid DataFrame, multiple simultaneous violations, empty DataFrame.",
            "Extend to also support regex pattern validation for string columns.",
        ],
    },
    {
        "root": "Write a function to convert a DataFrame column containing JSON strings into separate columns, handling malformed JSON by logging the row index and filling with NaN.",
        "followups": [
            "The JSON has deeply nested objects. Add a max_depth parameter to control how deep to flatten.",
            "Some rows have JSON arrays instead of objects. Handle them by creating list-type columns.",
            "Write unit tests for: valid JSON, malformed JSON, null values, nested JSON objects.",
            "Add a prefix parameter to avoid collisions between extracted columns and existing DataFrame columns.",
        ],
    },
    {
        "root": "Create a function to apply forward-fill, backward-fill, and linear interpolation to time series gaps, selecting the method automatically based on gap length.",
        "followups": [
            "The function modifies the DataFrame in place unexpectedly. Fix it to always return a new DataFrame.",
            "Add support for limit parameter to cap the maximum number of consecutive NaNs to fill.",
            "Write unit tests for: no gaps, single gap, gap at start, gap at end, entire column NaN.",
            "Extend to produce a visualization showing filled vs. original values for each column.",
        ],
    },
    {
        "root": "Write a pandas function to one-hot encode multiple categorical columns simultaneously, using a configurable unknown category handler for inference-time unseen values.",
        "followups": [
            "The encoded DataFrame has 500+ columns and causes memory issues downstream. Add a max_categories parameter to group rare values into 'other'.",
            "I'm getting a KeyError when a test set has a category not seen in training. Fix the unknown handler logic.",
            "Write unit tests for: single column, multiple columns, unseen category at inference, NaN values.",
            "Add a drop_first option for each column to avoid the dummy variable trap.",
        ],
    },
    {
        "root": "Implement a function to downcast DataFrame dtypes automatically: integers to the smallest int type that fits, floats to float32, and object columns with low cardinality to category.",
        "followups": [
            "Downcasting integer columns is causing overflow for values near int32 max. Add a safety check before downcasting.",
            "Add a report showing memory usage before and after and the dtype changes made.",
            "Write unit tests for: large integers, float precision loss check, high-cardinality vs low-cardinality strings.",
            "Extend to also handle Timestamp columns by downcasting to lower datetime resolution where safe.",
        ],
    },
    # ── Data Analysis ────────────────────────────────────────────────────────────────────────
    {
        "root": "Write a function to compute group-level aggregations using pandas groupby with multiple configurable aggregation functions per column, returning a flat DataFrame with readable column names.",
        "followups": [
            "The aggregation is slow on 50M rows. Profile it and rewrite the bottleneck using numpy groupby operations.",
            "I need to add a weighted mean aggregation that doesn't exist in standard pandas. Implement it as a custom agg function.",
            "Write unit tests for: empty group, single-row group, NaN values in aggregated column.",
            "Extend to also compute the contribution of each group to the total (share %) for numeric columns.",
        ],
    },
    {
        "root": "Create a function to compute a Pearson correlation matrix for numeric columns, identify pairs with absolute correlation above a threshold, and return a ranked summary table.",
        "followups": [
            "Add Spearman and Kendall options and let the caller choose the correlation method.",
            "The summary table includes both (A,B) and (B,A) as separate rows. Deduplicate to show each pair once.",
            "Write unit tests for: no high correlations, perfect correlation, single column, constant column.",
            "Extend to produce a heatmap visualization of the correlation matrix with the threshold pairs highlighted.",
        ],
    },
    {
        "root": "Implement a function to compute rolling statistics (mean, std, min, max) on a time series DataFrame with configurable window size, min_periods, and center alignment options.",
        "followups": [
            "The rolling computation produces NaN for the first window-1 rows. Add a fallback to use expanding window for those rows.",
            "I need exponentially weighted moving statistics instead of uniform rolling. Add an EWM option.",
            "Write unit tests for: window larger than series length, window=1, time series with gaps.",
            "Extend to compute rolling statistics grouped by an entity column for panel time series data.",
        ],
    },
    {
        "root": "Write a function to perform a chi-squared independence test between all pairs of categorical columns in a DataFrame, returning a p-value matrix and flagging significant pairs.",
        "followups": [
            "Some column pairs have cells with expected frequency below 5, violating chi-squared assumptions. Add a Fisher's exact test fallback for those cases.",
            "The Bonferroni correction for multiple comparisons isn't applied. Add multiple testing correction using statsmodels.",
            "Write unit tests for: independent columns, perfectly dependent columns, columns with single unique value.",
            "Extend to also compute Cramér's V effect size for each significant pair.",
        ],
    },
    {
        "root": "Create a function to compute cohort retention rates from user event data, producing a retention matrix indexed by cohort month and showing day-N retention percentages.",
        "followups": [
            "The cohort sizes are very different, making comparisons unfair. Normalize each row to percentage and add absolute counts as a second table.",
            "I need weekly cohorts instead of monthly. Make the cohort period configurable.",
            "Write unit tests for: single cohort, cohort with 100% retention, empty event log.",
            "Extend to produce a heatmap of the retention matrix with color coding by retention rate.",
        ],
    },
    {
        "root": "Write a function to perform A/B test analysis returning the p-value, effect size (Cohen's d), 95% confidence interval, and a verdict on statistical significance at a configurable alpha.",
        "followups": [
            "The test assumes normality but my metric is a conversion rate (binary). Switch to a proportion z-test for binary outcomes.",
            "Add a power analysis component that tells the caller how many samples were needed to detect the observed effect.",
            "Write unit tests for: clearly significant result, clearly not significant, equal sample sizes, very unequal sample sizes.",
            "Extend to support a Bayesian A/B test using beta-binomial conjugate posteriors as an alternative.",
        ],
    },
    {
        "root": "Implement a pivot table function that supports multiple simultaneous aggregation functions, handles missing group combinations with a configurable fill value, and returns a flat or hierarchical DataFrame.",
        "followups": [
            "Add a margins=True equivalent that computes row and column totals for each aggregation function.",
            "The output has a MultiIndex columns which is hard to work with. Flatten the column names automatically.",
            "Write unit tests for: single aggregation, multiple aggregations, missing combinations, empty input.",
            "Extend to support percentage-of-total normalization as an additional aggregation option.",
        ],
    },
    {
        "root": "Write a function to compute the distribution of values in each column and flag columns that deviate significantly from a provided reference distribution using a KS test.",
        "followups": [
            "The KS test is too sensitive for large samples. Add a PSI (Population Stability Index) computation as an alternative.",
            "Some columns are categorical, not continuous. Add a chi-squared based comparison for categorical columns.",
            "Write unit tests for: identical distributions, completely different distributions, constant column.",
            "Extend to produce a monitoring report suitable for use in a data pipeline health check.",
        ],
    },
    {
        "root": "Create a function to detect autocorrelation in a time series using ACF and PACF plots and suggest AR and MA orders based on the plots.",
        "followups": [
            "The suggestion logic is too simplistic. Implement AIC/BIC-based ARIMA order selection using statsmodels auto_arima.",
            "My time series has a non-stationary trend. Add an ADF test and recommend differencing before the ACF/PACF analysis.",
            "Write unit tests for: white noise series, AR(1) process, MA(1) process, constant series.",
            "Extend to also decompose the series into trend, seasonal, and residual components before the autocorrelation analysis.",
        ],
    },
    {
        "root": "Write a function to compute rank-based statistics — percentile rank, decile, and quintile — for each row within its group using pandas groupby and transform.",
        "followups": [
            "Ties are being ranked inconsistently. Add a tie-breaking strategy parameter (min, max, average, first).",
            "I need the ranks to be based on a rolling 90-day window within each group rather than the full history. Adapt for this.",
            "Write unit tests for: single group, multiple groups with ties, group with one row.",
            "Extend to also add a within-group z-score normalization column alongside the rank columns.",
        ],
    },
    # ── Feature Engineering ──────────────────────────────────────────────────────────────────
    {
        "root": "Write a function to extract temporal features from a datetime column: hour, day of week, month, quarter, is_weekend, days_since_epoch, and cyclic sine/cosine encodings of hour and day.",
        "followups": [
            "Add timezone-aware feature extraction that converts all timestamps to UTC before computing features.",
            "The cyclic encodings aren't normalized. Ensure they're in the range [-1, 1] for all period lengths.",
            "Write unit tests for: midnight edge case, leap year February 29, missing datetime values.",
            "Extend to also compute business day features: is_holiday (given a holiday calendar) and days_to_next_holiday.",
        ],
    },
    {
        "root": "Create a function to generate polynomial interaction features up to degree 2 from selected numeric columns, filtering out near-constant features by variance threshold.",
        "followups": [
            "The feature matrix explodes to 10,000 columns. Add L1 feature selection after generation to keep only the top-k.",
            "I'm getting numerical overflow for columns with large values. Add automatic pre-scaling before generating interactions.",
            "Write unit tests for: two columns, three columns, single column, constant column filtering.",
            "Extend to make this a scikit-learn transformer class compatible with Pipeline.",
        ],
    },
    {
        "root": "Implement target encoding using leave-one-out smoothing with cross-validation folds to prevent target leakage, and expose a transform method for inference-time encoding.",
        "followups": [
            "The smoothing factor is hardcoded. Add a parameter and a cross-validated grid search to tune it automatically.",
            "At inference time there are unseen categories. Fall back to the global target mean for those and log a warning.",
            "Write unit tests for: single category, high-cardinality column, category seen in test but not train.",
            "Extend to support multi-class targets by computing one encoding per class.",
        ],
    },
    {
        "root": "Write a function to build lag features and rolling window features for a time series, with configurable lag periods and window sizes, respecting temporal ordering and group boundaries.",
        "followups": [
            "The lag features are leaking future data because the time series wasn't sorted first. Fix the ordering guarantee.",
            "Add support for irregular time series where the timestamp spacing is not uniform.",
            "Write unit tests for: lag larger than series length, single-row group, multiple groups.",
            "Extend to also compute percent-change and ratio features between the current value and lagged values.",
        ],
    },
    {
        "root": "Implement a custom scikit-learn transformer that applies Yeo-Johnson transformation to skewed columns identified by a configurable skewness threshold, leaving other columns unchanged.",
        "followups": [
            "The transformer breaks when applied to a DataFrame with categorical columns. Add dtype checking to skip non-numeric columns.",
            "I need to invert the transformation for interpretability. Implement an inverse_transform method.",
            "Write unit tests for: all columns skewed, no columns skewed, NaN values, single column.",
            "Extend to also support Box-Cox transformation as an alternative and let the caller choose.",
        ],
    },
    {
        "root": "Create a function to compute aggregated features per entity — mean, std, count, min, max — and join them back to the original transaction-level DataFrame with prefixed column names.",
        "followups": [
            "The join is producing a Cartesian product on some entities. Debug the key and fix the merge.",
            "Add a configurable time window so only transactions within the last N days are used for aggregation.",
            "Write unit tests for: entity with single transaction, entity with all-NaN values, new entity at inference time.",
            "Extend to also compute entropy and mode for categorical columns per entity.",
        ],
    },
    {
        "root": "Write a function to create cyclic encodings for hour-of-day, day-of-week, and month-of-year features using sine and cosine transformations.",
        "followups": [
            "The encoding is applied to raw integer columns instead of properly scaled period fractions. Fix the normalization.",
            "Add an option to also keep the original integer column alongside the cyclic encoding.",
            "Write unit tests for: hour=0, hour=23, day=0, day=6, month=1, month=12.",
            "Extend to support any arbitrary cyclic feature with a configurable period length.",
        ],
    },
    {
        "root": "Implement a feature selection pipeline using mutual information scores, variance threshold, and a top-k selector, with a fit/transform interface compatible with scikit-learn Pipeline.",
        "followups": [
            "The mutual information scores change between runs. Fix the random seed handling for reproducibility.",
            "Add a minimum mutual information threshold as an alternative to fixed top-k selection.",
            "Write unit tests for: k larger than number of features, zero-variance column, binary target.",
            "Extend to also include a correlation-based redundancy filter that removes one of each highly correlated pair.",
        ],
    },
    {
        "root": "Write a function to compute entity-level frequency encoding for high-cardinality categorical variables with additive smoothing to handle rare categories.",
        "followups": [
            "The smoothing is treating unseen categories at inference as frequency zero. Replace with a configurable floor value.",
            "Add log-frequency encoding as an option alongside the raw frequency.",
            "Write unit tests for: single occurrence category, high-frequency category, unseen category at inference.",
            "Extend to expose a DataFrame with category, count, and encoded_value columns for inspection.",
        ],
    },
    {
        "root": "Create a pandas-based window feature generator for irregular time series that computes aggregations over time-based windows (e.g., last 7 days) rather than fixed row counts.",
        "followups": [
            "The rolling time window is not respecting group boundaries — it's mixing events from different entities. Fix the groupby logic.",
            "The window computation is O(n²) and too slow for 10M rows. Rewrite using merge_asof for efficiency.",
            "Write unit tests for: events within window, no events in window, window spanning group boundary.",
            "Extend to support multiple window sizes simultaneously and return all as new columns.",
        ],
    },
    # ── Supervised ML ────────────────────────────────────────────────────────────────────────
    {
        "root": "Build a scikit-learn pipeline for binary classification with imputation, standard scaling, one-hot encoding, and a gradient boosting classifier, evaluated with 5-fold stratified cross-validation.",
        "followups": [
            "The pipeline is leaking validation data through the imputer — fix by ensuring fit only happens on training folds.",
            "Add SHAP value computation after fitting to explain individual predictions.",
            "Write unit tests for: all-NaN column, single training sample, perfectly separable classes.",
            "Extend to also support regression targets with a switch parameter.",
        ],
    },
    {
        "root": "Implement a GridSearchCV workflow to tune a Random Forest classifier over n_estimators, max_depth, and min_samples_split, with early stopping based on OOB score.",
        "followups": [
            "The grid search is taking too long. Switch to RandomizedSearchCV with a time budget and log all evaluated configurations.",
            "Add a learning curve plot after the best model is selected to diagnose bias vs variance.",
            "Write unit tests for: single hyperparameter combination, no improvement over baseline.",
            "Extend to automatically re-fit on the full training set and produce a feature importance plot.",
        ],
    },
    {
        "root": "Write a function to train an XGBoost classifier with early stopping on a validation set and compute SHAP feature importances for the top-20 features.",
        "followups": [
            "The SHAP computation is running out of memory for 500 features. Use the tree_fast approximation instead.",
            "Add a partial dependence plot for the top-3 most important features.",
            "Write unit tests for: early stopping triggered at epoch 1, all features equally important, binary vs multiclass target.",
            "Extend to also compute interaction SHAP values and report the top-5 feature pairs.",
        ],
    },
    {
        "root": "Implement stratified k-fold cross-validation from scratch using pandas and numpy, computing mean and std of accuracy, F1, and AUROC across folds.",
        "followups": [
            "The stratification is not preserving class ratios in edge cases with very rare classes. Fix the split logic.",
            "Add support for nested cross-validation to get an unbiased estimate of hyperparameter-tuned model performance.",
            "Write unit tests for: k=2, k equal to sample count (LOOCV), binary vs multiclass target.",
            "Extend to also compute confidence intervals for each metric using bootstrap resampling.",
        ],
    },
    {
        "root": "Build a LightGBM multi-class classifier pipeline with class-weight balancing and a per-class precision-recall report.",
        "followups": [
            "The class weights are computed incorrectly — they're proportional to frequency instead of inversely proportional. Fix it.",
            "Add micro-averaged and macro-averaged F1 scores to the report alongside per-class metrics.",
            "Write unit tests for: two classes, five classes, single sample per class.",
            "Extend to plot a normalized confusion matrix heatmap after evaluation.",
        ],
    },
    {
        "root": "Write a time-series cross-validation function using TimeSeriesSplit that respects temporal ordering, avoids future leakage, and supports a configurable gap between train and test.",
        "followups": [
            "The gap parameter isn't being applied correctly — some test rows are too close to the training cutoff. Fix the index arithmetic.",
            "Add purging logic to remove training rows that overlap in time with the test period.",
            "Write unit tests for: n_splits=1, gap larger than test set, time series with duplicate timestamps.",
            "Extend to also compute walk-forward validation where the model is re-fitted at each split.",
        ],
    },
    {
        "root": "Create a stacking ensemble using scikit-learn StackingClassifier with logistic regression as the meta-learner, and compare it to each base learner individually.",
        "followups": [
            "The out-of-fold predictions for the meta-learner are causing data leakage. Verify and fix the cross-val_predict usage.",
            "Add calibration of the meta-learner's probabilities using isotonic regression.",
            "Write unit tests for: single base learner, all base learners identical, base learner that always predicts one class.",
            "Extend to also support a blending approach using a holdout set instead of cross-validation.",
        ],
    },
    {
        "root": "Implement probability calibration using isotonic regression and plot calibration curves (reliability diagrams) before and after calibration.",
        "followups": [
            "Calibration is overfitting because the calibration set is too small. Add cross-validated calibration using CalibratedClassifierCV.",
            "Add Brier score and ECE (expected calibration error) as quantitative calibration metrics.",
            "Write unit tests for: perfectly calibrated model, completely miscalibrated model.",
            "Extend to also support Platt scaling as an alternative calibration method.",
        ],
    },
    {
        "root": "Write a pipeline to handle class imbalance using SMOTE oversampling inside cross-validation folds without leaking synthetic samples into the validation fold.",
        "followups": [
            "SMOTE is being applied to the full dataset before cross-validation, causing leakage. Wrap it in an imblearn Pipeline.",
            "Compare SMOTE with ADASYN and random undersampling using the same cross-validation setup.",
            "Write unit tests for: minority class with 1 sample (SMOTE will fail), balanced classes, binary vs multiclass.",
            "Extend to also apply class weights as a baseline comparison alongside SMOTE.",
        ],
    },
    {
        "root": "Build a function to perform recursive feature elimination with cross-validation (RFECV) and plot the cross-validated score as a function of the number of features selected.",
        "followups": [
            "RFECV is too slow for 1000 features. Add a step parameter to eliminate multiple features per iteration.",
            "The selected feature set changes between runs. Add a random seed and verify reproducibility.",
            "Write unit tests for: more features than samples, all features equally important, single feature.",
            "Extend to also report the ranking of all features, not just the selected subset.",
        ],
    },
    {
        "root": "Implement a Bayesian hyperparameter search using Optuna for a gradient boosting model with AUROC as the objective and a configurable trial budget.",
        "followups": [
            "Optuna is trying hyperparameter values outside valid ranges. Add constraints to the search space definition.",
            "Add pruning using Optuna's MedianPruner to stop unpromising trials early.",
            "Write unit tests for: n_trials=1, objective returning NaN, fixed seed for reproducibility.",
            "Extend to visualize the hyperparameter importance using Optuna's built-in plot functions.",
        ],
    },
    {
        "root": "Write a function to serialize a trained scikit-learn pipeline to disk using joblib and verify it produces byte-identical predictions after deserialization.",
        "followups": [
            "The deserialized pipeline fails on newer sklearn versions. Add version metadata and a compatibility check.",
            "Add a checksum of the model file to detect corruption before loading.",
            "Write unit tests for: pipeline with custom transformer, pipeline saved and loaded in different Python processes.",
            "Extend to also serialize and restore the normalization parameters and feature names.",
        ],
    },
    {
        "root": "Create a voting ensemble of three classifiers (SVM, Random Forest, Gradient Boosting) using both hard and soft voting, and compare their AUROC on a holdout set.",
        "followups": [
            "The SVM doesn't support predict_proba for soft voting. Wrap it with CalibratedClassifierCV to add probability estimates.",
            "Add weights to the soft voting based on each classifier's cross-validated AUROC.",
            "Write unit tests for: all classifiers agree, all classifiers disagree, one classifier always wrong.",
            "Extend to also include a diversity analysis measuring pairwise agreement between base classifiers.",
        ],
    },
    {
        "root": "Implement permutation feature importance analysis and compare it to the model's native feature importances for a Random Forest, highlighting disagreements.",
        "followups": [
            "Permutation importance is slow for 200 features and 100K samples. Reduce runtime using sampling.",
            "Add confidence intervals for permutation importance using repeated permutations.",
            "Write unit tests for: single feature, feature with zero importance, correlated features.",
            "Extend to also compute SHAP values and produce a three-way comparison plot.",
        ],
    },
    {
        "root": "Write a function to perform learning curve analysis to plot training and cross-validation scores vs. training set size, diagnosing whether the model is underfitting or overfitting.",
        "followups": [
            "The learning curves are noisy. Increase the number of cross-validation folds and add confidence interval shading.",
            "Add a recommendation engine that prints whether the model needs more data, more regularization, or a more complex model.",
            "Write unit tests for: linear model on non-linear data, perfectly fitting model, random baseline.",
            "Extend to also show the effect of increasing regularization strength as a separate plot.",
        ],
    },
    # ── Unsupervised ML ──────────────────────────────────────────────────────────────────────
    {
        "root": "Implement k-means clustering with the elbow method and silhouette score to select the optimal number of clusters, and visualize the cluster assignments in 2D using PCA.",
        "followups": [
            "K-means is producing empty clusters for k=8. Add a safeguard that retries with a different initialization.",
            "The elbow and silhouette methods suggest different k values. Add the Davies-Bouldin index as a tiebreaker.",
            "Write unit tests for: k=1, k equal to number of samples, all identical points.",
            "Extend to also compute cluster statistics (size, centroid, within-cluster variance) for the optimal k.",
        ],
    },
    {
        "root": "Write a function to perform PCA on a high-dimensional dataset, plot cumulative explained variance, and return the minimum number of components explaining a configurable variance threshold.",
        "followups": [
            "The PCA is not centering the data before decomposition, which produces incorrect results. Fix the preprocessing.",
            "Add a biplot visualization showing both sample projections and feature loadings.",
            "Write unit tests for: threshold=1.0, threshold=0.0, more features than samples.",
            "Extend to also compare PCA with kernel PCA and t-SNE on a 2D visualization.",
        ],
    },
    {
        "root": "Create a hierarchical clustering pipeline with Ward linkage, produce a labeled dendrogram, and cut the tree at the optimal number of clusters using the inconsistency coefficient.",
        "followups": [
            "The dendrogram labels are overlapping for 500 samples. Add a truncation option to show only the top levels.",
            "The inconsistency coefficient is not converging to a clear cut point. Add the Calinski-Harabasz score as an alternative.",
            "Write unit tests for: two clearly separated clusters, all points in one cluster, single-element clusters.",
            "Extend to compare Ward with complete and average linkage using the cophenetic correlation coefficient.",
        ],
    },
    {
        "root": "Implement DBSCAN clustering on a geospatial dataset with haversine distance metric, and visualize cluster assignments and noise points on a scatter plot.",
        "followups": [
            "The haversine distance metric is not supported directly by DBSCAN's metric parameter. Implement a custom distance function.",
            "Almost all points are classified as noise. Guide me on selecting eps and min_samples using a k-distance plot.",
            "Write unit tests for: all points in one cluster, all points noise, two clearly separated clusters.",
            "Extend to also compare DBSCAN results with HDBSCAN which handles varying density clusters better.",
        ],
    },
    {
        "root": "Build an anomaly detection pipeline using Isolation Forest and Local Outlier Factor, and compare their detection rates and false positive rates on a labeled anomaly dataset.",
        "followups": [
            "Both models are flagging 30% of data as anomalies because contamination is set too high. Add automatic contamination estimation.",
            "Combine both models using an ensemble: flag as anomaly only when both agree.",
            "Write unit tests for: no anomalies, all anomalies, contamination=0.5.",
            "Extend to also add a threshold-tuning curve showing precision and recall at different contamination levels.",
        ],
    },
    # ── Visualization ────────────────────────────────────────────────────────────────────────
    {
        "root": "Write a function to create a multi-panel matplotlib EDA figure: histogram with KDE for each numeric column, boxplot per class, and a correlation heatmap with significance masking.",
        "followups": [
            "The figure has too many subplots when there are 50 columns. Add pagination so figures never exceed a configurable max panels.",
            "The significance masking is using p < 0.05 with no multiple testing correction. Add Bonferroni correction.",
            "Write unit tests for: single column, all constant columns, no numeric columns.",
            "Extend to also add a pairplot for the top-5 most correlated feature pairs.",
        ],
    },
    {
        "root": "Create a seaborn visualization showing the joint distribution of two continuous features with a regression line, confidence band, and marginal histograms.",
        "followups": [
            "The regression line is heavily influenced by outliers. Add a robust regression option using Theil-Sen estimator.",
            "Add a color-coded hue dimension to show separate joint distributions per class.",
            "Write unit tests for: perfectly correlated features, zero correlation, features with different scales.",
            "Extend to also add the Pearson r and p-value as a text annotation on the plot.",
        ],
    },
    {
        "root": "Write a function to plot training and validation loss curves with confidence bands across multiple runs, annotating the point of minimum validation loss and potential overfitting region.",
        "followups": [
            "The confidence bands are computed incorrectly using full std instead of std/sqrt(n). Fix the standard error calculation.",
            "Add a smoothing option using exponential moving average to reduce noise in the curves.",
            "Write unit tests for: single run (no confidence band), loss that never converges, validation better than train.",
            "Extend to also support plotting multiple metrics side-by-side (e.g., loss and accuracy).",
        ],
    },
    {
        "root": "Build a function to generate an interactive Plotly time series dashboard with a date range slider, moving average overlay, and anomaly annotation.",
        "followups": [
            "The anomaly annotations are cluttering the chart when there are many anomalies. Add a toggle button to show/hide them.",
            "The moving average window isn't configurable in the interactive chart. Add a dropdown to select the window size.",
            "Write unit tests for: empty time series, time series with NaN, single data point.",
            "Extend to also add a forecast line using a simple ARIMA model alongside the historical data.",
        ],
    },
    {
        "root": "Implement a confusion matrix visualization showing normalized percentages, per-class precision and recall as bar charts below the matrix, and a title showing overall accuracy and F1.",
        "followups": [
            "The normalization is dividing by row totals but the label says 'column-normalized'. Fix either the calculation or the label.",
            "Add an option to display both raw counts and percentages simultaneously in each cell.",
            "Write unit tests for: binary classification, 10-class classification, zero-count class.",
            "Extend to also show the misclassification breakdown: for each true class, which predicted class received the most errors.",
        ],
    },
    # ── Performance & Scalability ────────────────────────────────────────────────────────────
    {
        "root": "Optimize a pandas groupby aggregation on a 10M-row DataFrame that is taking 45 seconds — profile it and rewrite the bottleneck.",
        "followups": [
            "After converting the group column to category dtype, the groupby is still slow. The bottleneck shifted to the agg step — profile and fix.",
            "Rewrite using numpy's group operations directly to bypass pandas overhead.",
            "Write unit tests to verify the optimized version produces identical output to the original.",
            "Extend the optimization to also work with Dask for datasets larger than RAM.",
        ],
    },
    {
        "root": "Write a function to process a CSV file that is too large to fit in RAM using pandas chunked reading with adaptive chunk sizing and a progress bar.",
        "followups": [
            "The adaptive chunk sizing is increasing the chunk size even when memory is already at 80%. Add a memory usage cap.",
            "Add parallel chunk processing using concurrent.futures while maintaining output order.",
            "Write unit tests for: file smaller than chunk size, file with 0 rows, last chunk smaller than chunk_size.",
            "Extend to support Parquet output alongside CSV for memory-efficient storage of results.",
        ],
    },
    {
        "root": "Implement parallel feature computation across DataFrame column groups using Python multiprocessing Pool with proper error handling, progress tracking, and graceful shutdown.",
        "followups": [
            "The multiprocessing Pool is spawning 32 workers on an 8-core machine, causing thrashing. Auto-detect CPU count.",
            "Errors in worker processes are being silently swallowed. Re-raise them in the main process with the original traceback.",
            "Write unit tests for: single column, error in one worker, pool shutdown during processing.",
            "Extend to use joblib as a backend with a Dask fallback for cluster environments.",
        ],
    },
    {
        "root": "Create a function to profile a pandas DataFrame memory usage column by column and produce a report with current dtype, suggested dtype, and estimated memory savings.",
        "followups": [
            "Downcasting a float64 column to float32 is causing precision loss that affects downstream calculations. Add a max_precision_loss parameter.",
            "Add a dry_run mode that returns the report without modifying the DataFrame.",
            "Write unit tests for: DataFrame with all optimal dtypes already, mixed dtypes, empty DataFrame.",
            "Extend to also visualize memory usage before and after as a bar chart.",
        ],
    },
    {
        "root": "Write a Dask-based pipeline to compute groupby aggregations on a 50GB dataset partitioned across multiple parquet files, with progress reporting.",
        "followups": [
            "Dask is loading all partitions into memory simultaneously instead of lazily. Fix the computation graph.",
            "The results differ from a pandas reference implementation on a small sample. Debug the discrepancy.",
            "Write unit tests that run on small in-memory Dask DataFrames to verify correctness.",
            "Extend to support writing the aggregation results back to a partitioned parquet dataset.",
        ],
    },
    # ── Statistical Analysis ─────────────────────────────────────────────────────────────────
    {
        "root": "Implement bootstrap resampling to compute 95% confidence intervals for mean, median, standard deviation, and a custom metric on a dataset with configurable number of bootstrap iterations.",
        "followups": [
            "The confidence interval for the median is asymmetric in a suspicious way. Verify the percentile vs. BCa interval methods.",
            "Bootstrap with 10,000 iterations is slow. Parallelize using numpy's vectorized sampling.",
            "Write unit tests for: n_iterations=1, sample of size 1, constant sample (zero variance).",
            "Extend to also compute bootstrap confidence intervals for the difference between two groups.",
        ],
    },
    {
        "root": "Write a function to perform two-sample hypothesis testing using both parametric (Welch's t-test) and non-parametric (Mann-Whitney U) methods, with effect size and power computation.",
        "followups": [
            "The power computation uses the wrong tail specification for a one-sided test. Fix the alternative hypothesis handling.",
            "Add a normality check using Shapiro-Wilk and automatically select the appropriate test.",
            "Write unit tests for: identical distributions, completely non-overlapping distributions, one sample of size 1.",
            "Extend to also handle paired samples using the paired t-test and Wilcoxon signed-rank test.",
        ],
    },
    {
        "root": "Create a function to fit a time series decomposition model using statsmodels seasonal_decompose, and plot trend, seasonal, and residual components with anomaly detection on residuals.",
        "followups": [
            "The seasonal_decompose is returning NaN at the edges due to the centered moving average. Switch to STL decomposition.",
            "The anomaly detection threshold on residuals is too sensitive. Add a configurable sigma multiplier.",
            "Write unit tests for: trend-only series, seasonal-only series, pure noise.",
            "Extend to also forecast the next N periods by extrapolating the trend and adding the seasonal component.",
        ],
    },
    {
        "root": "Write a function to perform survival analysis using the Kaplan-Meier estimator, compare two groups with the log-rank test, and produce a survival curve plot with confidence intervals.",
        "followups": [
            "The confidence intervals are computed using Greenwood's formula but the plot is showing standard error bands instead. Fix the formula.",
            "Add a median survival time annotation with confidence interval to the plot.",
            "Write unit tests for: 100% event rate, 0% event rate, single observation.",
            "Extend to also fit a Cox proportional hazards model and report the hazard ratio.",
        ],
    },
    {
        "root": "Implement a function to detect structural breaks in a time series using the CUSUM (cumulative sum) method and return the timestamps and magnitudes of detected breakpoints.",
        "followups": [
            "The CUSUM threshold is hardcoded. Make it adaptive based on the series standard deviation.",
            "Add the Chow test as an alternative breakpoint detection method for comparison.",
            "Write unit tests for: no breakpoints, single breakpoint in the middle, breakpoint at the start.",
            "Extend to also visualize the CUSUM statistic alongside the original series with breakpoints marked.",
        ],
    },
    # ── NLP and Text ─────────────────────────────────────────────────────────────────────────
    {
        "root": "Build a scikit-learn text classification pipeline using TF-IDF vectorization and a logistic regression classifier, evaluated with 5-fold cross-validation on a multi-class dataset.",
        "followups": [
            "The TF-IDF vocabulary is too large. Add a max_features limit and compare performance before and after.",
            "Add a bigram and trigram feature option and measure whether it improves accuracy.",
            "Write unit tests for: empty document, all-unknown vocabulary at inference, single-class dataset.",
            "Extend the pipeline to also support sentence embeddings using a pre-trained spaCy model.",
        ],
    },
    {
        "root": "Write a function to preprocess a corpus of text documents: tokenize, remove stopwords, apply stemming, and build a bag-of-words representation as a sparse matrix.",
        "followups": [
            "The stopword list is English-only. Add multilingual stopword support using NLTK's corpus.",
            "Stemming is changing 'data' to 'dat'. Switch to lemmatization using spaCy and compare token quality.",
            "Write unit tests for: empty corpus, document with only stopwords, document with special characters.",
            "Extend to also extract named entities and add entity type frequency as additional features.",
        ],
    },
    {
        "root": "Implement a topic modeling pipeline using Latent Dirichlet Allocation on a document corpus, selecting the optimal number of topics using coherence score.",
        "followups": [
            "The coherence score is not converging — the vocabulary is too large. Add a minimum document frequency filter.",
            "Add pyLDAvis visualization to interactively explore topic-term distributions.",
            "Write unit tests for: n_topics=1, corpus with 2 documents, corpus with highly similar documents.",
            "Extend to assign a dominant topic label to each document and produce a topic distribution DataFrame.",
        ],
    },
    # ── Recommendation and Ranking ────────────────────────────────────────────────────────────
    {
        "root": "Build a collaborative filtering recommendation system using matrix factorization with SVD on a user-item rating matrix, and evaluate with RMSE on a holdout set.",
        "followups": [
            "The SVD can't handle new users at inference time (cold start). Add a popularity-based fallback for them.",
            "The rating matrix is 99% sparse and SVD is slow. Switch to Alternating Least Squares (ALS) with scipy.sparse.",
            "Write unit tests for: user with no ratings, item with no ratings, all-zero matrix.",
            "Extend to also implement item-based collaborative filtering and compare RMSE with the SVD approach.",
        ],
    },
    {
        "root": "Implement a content-based recommendation engine that ranks items by cosine similarity to a user profile vector built from their interaction history.",
        "followups": [
            "Items with many features dominate the similarity score. Normalize item vectors before computing cosine similarity.",
            "The recommendation list includes items the user already interacted with. Add a filter to exclude seen items.",
            "Write unit tests for: zero-vector user profile, single-item catalogue, user with interactions on all items.",
            "Extend to implement a hybrid model that blends content-based and collaborative scores with a configurable weight.",
        ],
    },
    # ── Time Series Forecasting ───────────────────────────────────────────────────────────────
    {
        "root": "Write a function to fit a SARIMA model to a monthly time series with seasonal period 12, selecting p, d, q orders using AIC minimization over a grid search.",
        "followups": [
            "The AIC grid search is too slow for large parameter grids. Switch to auto_arima from pmdarima.",
            "The residuals show remaining autocorrelation. Diagnose using a Ljung-Box test and adjust the model.",
            "Write unit tests for: non-seasonal series, series shorter than 2 seasonal periods.",
            "Extend to produce a forecast plot with confidence intervals overlaid on the training history.",
        ],
    },
    {
        "root": "Implement a Prophet-based forecasting pipeline that fits a daily time series with weekly and yearly seasonality, adds custom holiday effects, and produces 90-day forecasts.",
        "followups": [
            "Prophet is producing negative forecasts for a strictly positive metric. Add a log transformation pre- and post-fit.",
            "The holiday effects are too large and dominating the forecast. Add a prior_scale parameter to dampen them.",
            "Write unit tests for: time series with gaps, series with zero variation, future dates beyond training range.",
            "Extend to cross-validate the Prophet model using its built-in cross_validation function and compute MAE and MAPE.",
        ],
    },
    # ── Model Explainability ──────────────────────────────────────────────────────────────────
    {
        "root": "Implement LIME (Local Interpretable Model-agnostic Explanations) for explaining individual predictions of a black-box classifier on tabular data.",
        "followups": [
            "The LIME explanations are inconsistent across runs due to random sampling. Fix the random seed for reproducibility.",
            "The explanations for negative class predictions are confusing. Add a target parameter to always explain the positive class.",
            "Write unit tests for: single-feature dataset, binary vs multiclass target.",
            "Extend to also compute global feature importance by averaging LIME weights across a sample of predictions.",
        ],
    },
    {
        "root": "Write a function to compute and visualize SHAP (SHapley Additive exPlanations) values for a trained gradient boosting model, producing a summary plot and a waterfall plot for a single prediction.",
        "followups": [
            "The SHAP computation runs out of memory for 1M rows. Switch to approximate TreeSHAP with a sample.",
            "Add a dependence plot for the top feature showing its interaction with the most correlated feature.",
            "Write unit tests for: model trained on 1 feature, model predicting all one class.",
            "Extend to also compute SHAP interaction values and display the top-5 pairwise interactions.",
        ],
    },
    # ── Geospatial ────────────────────────────────────────────────────────────────────────────
    {
        "root": "Write a function to cluster geographic coordinates (lat/lon) using DBSCAN with haversine distance metric and visualize the clusters on a Folium map.",
        "followups": [
            "Points near the antimeridian are being assigned to incorrect clusters. Fix the coordinate wrapping issue.",
            "Add an option to also compute the centroid of each cluster and annotate them on the map.",
            "Write unit tests for: two clearly separated location clusters, all points within 1 meter.",
            "Extend to also compute the convex hull of each cluster and overlay it as a polygon on the map.",
        ],
    },
    {
        "root": "Create a function to compute the distance between each pair of points in a geospatial DataFrame using vectorized haversine formula, and return the nearest neighbor for each point.",
        "followups": [
            "The pairwise computation is O(n²) and too slow for 100K points. Switch to a KD-tree or ball-tree approach.",
            "The haversine result is in kilometers but I need miles. Add a unit parameter.",
            "Write unit tests for: two identical coordinates, antipodal points, single-point DataFrame.",
            "Extend to also find all points within a configurable radius for each point.",
        ],
    },
    # ── MLOps and Pipelines ───────────────────────────────────────────────────────────────────
    {
        "root": "Set up an MLflow experiment tracking workflow that logs parameters, metrics, and artifacts for a scikit-learn model training run.",
        "followups": [
            "The MLflow server isn't available in the test environment. Add a fallback to file-based logging.",
            "The artifact logging is too slow because it serializes the model on each training fold. Log only the final model.",
            "Write unit tests mocking the MLflow client to verify the correct parameters and metrics are logged.",
            "Extend to also register the best model to the MLflow Model Registry and transition it to staging.",
        ],
    },
    {
        "root": "Build a scikit-learn Pipeline with preprocessing, feature selection, and a classifier, then export it to ONNX format for language-agnostic inference.",
        "followups": [
            "The ONNX export fails because of a custom transformer. Implement the required __sklearn_tags__ interface.",
            "Verify that ONNX inference produces identical predictions to sklearn inference within floating-point tolerance.",
            "Write unit tests comparing sklearn and ONNX output on a held-out test set.",
            "Extend to benchmark the ONNX inference speed against the sklearn pipeline.",
        ],
    },
    # ── Imbalanced Learning ───────────────────────────────────────────────────────────────────
    {
        "root": "Implement a threshold optimization function that selects the decision threshold for a binary classifier maximizing F1 score on a validation set.",
        "followups": [
            "F1 is the wrong metric here — the business cost of false negatives is 10× false positives. Optimize the threshold for minimum cost instead.",
            "The optimal threshold on the validation set doesn't generalize. Add cross-validated threshold selection.",
            "Write unit tests for: all predictions at 0.5, perfect classifier, worst possible classifier.",
            "Extend to plot a threshold vs. precision/recall/F1 curve to visualize the trade-offs.",
        ],
    },
    {
        "root": "Write a function to evaluate a classifier on a highly imbalanced dataset using precision-recall AUC, average precision, and the Matthews correlation coefficient.",
        "followups": [
            "The AUC-ROC is misleadingly high because of the class imbalance. Add a note to the report explaining why AUPRC is preferred.",
            "Add a comparison of all metrics at the default 0.5 threshold vs. the optimized threshold.",
            "Write unit tests for: 1% positive rate, 50% positive rate, classifier that predicts all negative.",
            "Extend to also compute and display the lift curve and the gain chart.",
        ],
    },
    # ── Graph Data ────────────────────────────────────────────────────────────────────────────
    {
        "root": "Write a function using NetworkX to build a directed graph from a transaction DataFrame, compute PageRank for each node, and return the top-20 most influential nodes.",
        "followups": [
            "PageRank is not converging. Increase max_iter and add a convergence check with tolerance logging.",
            "The graph has 10M edges and NetworkX runs out of memory. Switch to a sparse matrix PageRank implementation.",
            "Write unit tests for: single-node graph, fully connected graph, disconnected graph.",
            "Extend to also compute betweenness centrality and compare rankings with PageRank.",
        ],
    },
    {
        "root": "Implement a community detection pipeline on a social network graph using the Louvain algorithm and visualize the communities with distinct colors.",
        "followups": [
            "The Louvain result changes on every run. Fix the random seed for reproducibility.",
            "Add a modularity score to quantify the quality of the detected communities.",
            "Write unit tests for: two cliques connected by one edge, fully random graph, single node.",
            "Extend to also compute the inter-community edge density as a measure of community boundary strength.",
        ],
    },
    # ── Multi-label and Multi-output ─────────────────────────────────────────────────────────
    {
        "root": "Build a multi-label classification pipeline where each sample can belong to multiple classes, evaluated with Hamming loss, subset accuracy, and per-label F1.",
        "followups": [
            "Some labels are extremely rare. Add per-label class weighting to the loss function.",
            "The predict_proba output shape is wrong after OneVsRestClassifier wrapping. Debug and fix the shape mismatch.",
            "Write unit tests for: all labels absent, all labels present, single-label sample.",
            "Extend to also implement label correlation analysis and visualize co-occurrence frequency.",
        ],
    },
    # ── Data Quality and Monitoring ───────────────────────────────────────────────────────────
    {
        "root": "Implement a data drift detection function that compares feature distributions between a reference dataset and a current batch using PSI and KS tests, flagging drifted columns.",
        "followups": [
            "PSI bins are collapsing because some reference bins have zero count. Add smoothing to prevent division by zero.",
            "The drift report is hard to act on. Add a severity classification: no drift, minor drift, severe drift.",
            "Write unit tests for: identical distributions, completely different distributions, single-value column.",
            "Extend to also detect schema drift (new columns, missing columns, dtype changes) alongside distribution drift.",
        ],
    },
    {
        "root": "Write a Great Expectations-based data validation suite that checks non-null constraints, value ranges, and referential integrity for a daily data pipeline.",
        "followups": [
            "Great Expectations is throwing a deprecation warning for ExpectationSuite. Upgrade to the v3 API.",
            "Add a custom expectation that verifies a column's values follow a specific regex pattern.",
            "Write unit tests for: all expectations pass, single expectation fails, empty DataFrame.",
            "Extend to also send a Slack notification when validation fails using a webhook.",
        ],
    },
    # ── Additional Python DS sessions to reach 100 ────────────────────────────────────────────
    {
        "root": "Write a scikit-learn pipeline that applies PCA after scaling and feeds the components into a SVM classifier, tuning the number of components and SVM C parameter jointly.",
        "followups": [
            "The PCA is applied before the cross-validation outer loop, leaking validation data. Fix the pipeline structure.",
            "Add a whitening option to PCA and measure its impact on SVM accuracy.",
            "Write unit tests for: n_components larger than n_features, C=0 SVM, all identical samples.",
            "Extend to compare PCA+SVM with kernel PCA+SVM and plot the decision boundary.",
        ],
    },
    {
        "root": "Create a function to compute Mutual Information between each feature and a target variable using scikit-learn, returning a ranked DataFrame with MI scores and p-values.",
        "followups": [
            "The MI scores are sensitive to the number of neighbors. Add a k_neighbors parameter and show the sensitivity.",
            "Add a comparison with Pearson correlation and Spearman rank correlation for the same feature set.",
            "Write unit tests for: constant feature, perfectly predictive feature, binary vs continuous target.",
            "Extend to plot MI scores as a horizontal bar chart with error bars from repeated estimation.",
        ],
    },
    {
        "root": "Build a pandas-based ETL pipeline that reads from multiple sources, applies transformations, and writes to a partitioned Parquet dataset with schema validation at each step.",
        "followups": [
            "The schema validation is happening after the transformation. Move it to a pre- and post-step check.",
            "Add checkpoint logic that saves intermediate results so the pipeline can resume after failure.",
            "Write unit tests for: source with missing schema field, transformation that produces empty output.",
            "Extend to support delta-loading by processing only rows newer than the last run timestamp.",
        ],
    },
    {
        "root": "Implement a hyperparameter sensitivity analysis function that varies one parameter at a time, measuring the effect on AUROC while holding all others at their best values.",
        "followups": [
            "The single-variable sweep misses interaction effects. Add a pairwise sweep for the top-2 most sensitive parameters.",
            "Add visualization of the sensitivity curve for each parameter with confidence intervals.",
            "Write unit tests for: single hyperparameter, parameter with no effect, discrete vs continuous parameter.",
            "Extend to compute a global sensitivity index using Sobol' variance decomposition.",
        ],
    },
    {
        "root": "Write a function to compute and compare the training and inference throughput of multiple scikit-learn models, reporting samples per second and memory usage.",
        "followups": [
            "The memory measurement is capturing the full process memory. Use tracemalloc to measure only the model's allocation.",
            "Add a batch-size sensitivity analysis showing how throughput changes with larger inference batches.",
            "Write unit tests for: model that returns immediately, model that errors during prediction.",
            "Extend to also benchmark serialization and deserialization speed for each model.",
        ],
    },
    {
        "root": "Create a pandas function to compute a feature interaction matrix showing the joint entropy between all pairs of categorical features in a DataFrame.",
        "followups": [
            "The entropy computation is in nats but I need bits. Convert to log base 2.",
            "Add mutual information (joint entropy minus marginal entropies) as a derived metric.",
            "Write unit tests for: fully independent features, perfectly correlated features, single unique value.",
            "Extend to visualize the matrix as a heatmap with hierarchical clustering of features.",
        ],
    },
    {
        "root": "Write a function to perform stratified train/test split at the group level (e.g., by patient ID) ensuring no group appears in both train and test sets.",
        "followups": [
            "The stratification is based on sample-level labels, not group-level. Fix to compute group-level label distribution.",
            "Some groups are too large and dominate the test set. Add a max-group-size cap with subsampling.",
            "Write unit tests for: single group, more groups than samples, all samples in same group.",
            "Extend to support k-fold group cross-validation preserving group boundaries.",
        ],
    },
    {
        "root": "Implement a function that applies random search over a scikit-learn pipeline, caching preprocessing steps between iterations to avoid redundant computation.",
        "followups": [
            "The cache is not invalidated when the pipeline steps change. Add a cache key based on the pipeline config hash.",
            "Switch from random search to halving random search to reduce total compute time.",
            "Write unit tests for: cache hit on second identical trial, cache miss after step change.",
            "Extend to parallelize trials using joblib and verify the cache is thread-safe.",
        ],
    },
    {
        "root": "Write a function to detect and report class label noise in a training dataset using cross-validated prediction confidence as a proxy for mislabeling probability.",
        "followups": [
            "The confidence threshold is hardcoded. Make it adaptive based on the per-class distribution of scores.",
            "Add a comparison with the cleanlab library's confident learning approach.",
            "Write unit tests for: clean dataset, fully noisy labels, single-class dataset.",
            "Extend to also output a corrected version of the dataset with suspected mislabels removed.",
        ],
    },
    {
        "root": "Create a pandas function to compute a co-occurrence matrix from a column containing lists of items, normalize it by item frequency, and return the top-k co-occurring pairs.",
        "followups": [
            "The normalization is producing NaN for items that appear with no other items. Handle zero-frequency items.",
            "The computation is O(n²) for large item sets. Rewrite using sparse matrix operations.",
            "Write unit tests for: empty lists, lists with single items, items appearing in all lists.",
            "Extend to also visualize the top-k pairs as a network graph using NetworkX.",
        ],
    },
]

# ── Adversarial code themes: 200 root tasks per theme, shared follow-up pool ─────────────────

ADVERSARIAL_FOLLOWUP_POOL = {
    "javascript": [
        "Add comprehensive error handling with try-catch blocks and meaningful error messages.",
        "I'm getting a TypeError: cannot read property of undefined. Debug it and add null checks throughout.",
        "Write Jest unit tests covering happy path, edge cases, and error conditions.",
        "Refactor using modern ES2022+ syntax and ensure compatibility with TypeScript strict mode.",
        "Optimize this for performance — profile and fix the bottleneck.",
        "Add JSDoc comments and TypeScript type definitions for all exported functions.",
        "Extend this to handle asynchronous operations using async/await with proper error propagation.",
        "Add input validation and sanitization to prevent XSS and injection vulnerabilities.",
    ],
    "cpp": [
        "Add RAII wrappers and ensure no memory leaks using Valgrind-clean code.",
        "I'm getting a segmentation fault. Add bounds checking and fix the memory access violation.",
        "Write Google Test unit tests covering construction, normal operation, and edge cases.",
        "Refactor using modern C++17 features: std::optional, structured bindings, and if-constexpr.",
        "Optimize the hot path using SIMD intrinsics or cache-friendly data layout.",
        "Add const-correctness, noexcept specifications, and move semantics throughout.",
        "Extend this with a thread-safe version using std::mutex or std::atomic where appropriate.",
        "Add static analysis compliance: fix all Clang-Tidy and cppcheck warnings.",
    ],
    "sql": [
        "Optimize the query using appropriate indexes — show the EXPLAIN ANALYZE output and fix the sequential scan.",
        "I'm getting a deadlock error under concurrent load. Rewrite to avoid the deadlock condition.",
        "Write test cases for the query covering empty table, single row, and boundary values.",
        "Refactor to use a CTE instead of nested subqueries for readability.",
        "Add proper transaction handling with ROLLBACK on error.",
        "Rewrite this as a stored procedure with input validation and error handling.",
        "Extend to also handle NULL values correctly in all join and filter conditions.",
        "Add a partitioning strategy to improve query performance on large tables.",
    ],
    "java": [
        "Add proper exception handling with custom exception classes and meaningful messages.",
        "I'm getting a NullPointerException at runtime. Add null checks and use Optional where appropriate.",
        "Write JUnit 5 tests covering normal operation, boundary values, and exception cases.",
        "Refactor using Java Stream API and functional interfaces to eliminate explicit loops.",
        "Add generics to make this type-safe and remove unchecked cast warnings.",
        "Implement the Builder pattern for object construction and add Javadoc documentation.",
        "Extend with Spring Boot annotations and make it a proper REST endpoint with validation.",
        "Add thread safety using synchronized blocks or java.util.concurrent classes.",
    ],
    "html_css": [
        "Make this fully responsive for mobile, tablet, and desktop using CSS Grid and Flexbox.",
        "I'm seeing a layout break in Safari. Fix the cross-browser compatibility issues.",
        "Add WCAG 2.1 AA accessibility: ARIA roles, keyboard navigation, and sufficient color contrast.",
        "Refactor the CSS using custom properties (CSS variables) for theming.",
        "Optimize for Core Web Vitals: reduce layout shift and improve paint performance.",
        "Add CSS animations and transitions with prefers-reduced-motion media query support.",
        "Extend the component to support a dark mode using the prefers-color-scheme media query.",
        "Add print styles to ensure the layout prints correctly on A4 paper.",
    ],
}

ADVERSARIAL_THEMES = {
    "javascript": {
        "topic": "javascript",
        "followup_pool": ADVERSARIAL_FOLLOWUP_POOL["javascript"],
        "root_tasks": [
            # React Components
            "Build a React functional component for a paginated data table with column sorting and keyword filtering.",
            "Create a React form with real-time field validation displaying inline error messages using controlled components.",
            "Write a React custom hook that fetches paginated data from an API with loading and error states.",
            "Implement a React drag-and-drop list using the HTML5 Drag and Drop API without external libraries.",
            "Build a React autocomplete input component with debounced API search and keyboard navigation.",
            "Create a React modal dialog with focus trapping, ESC key dismissal, and backdrop click handling.",
            "Write a React data visualization component that renders a bar chart using D3.js inside a useEffect.",
            "Implement a React multi-step form wizard with progress indicator and per-step validation.",
            "Build a React infinite scroll component that loads more items as the user scrolls near the bottom.",
            "Create a React context provider for global theme management with light and dark mode toggling.",
            "Write a React component that displays a real-time countdown timer with start, pause, and reset controls.",
            "Implement a React image gallery with lazy loading, zoom on click, and keyboard navigation.",
            "Build a React table component with resizable columns, row selection checkboxes, and a bulk action toolbar.",
            "Create a React date range picker with a calendar popup and formatted date display.",
            "Write a React tree view component supporting nested data, expand/collapse nodes, and node selection.",
            "Implement a React file upload component with drag-and-drop, progress bar, and file type validation.",
            "Build a React rich text editor wrapper around Quill.js with configurable toolbar options.",
            "Create a React virtualized list for efficiently rendering 100,000 rows using react-window.",
            "Write a React toast notification system with auto-dismiss timer and notification queue.",
            "Implement a React card grid that adapts column count to the viewport width.",
            "Build a React search results page with sidebar filters and URL parameter synchronization.",
            "Create a React kanban board with draggable cards across columns using react-beautiful-dnd.",
            "Write a React accordion component with animated expand and collapse transitions.",
            "Implement a React command palette with fuzzy search across a list of commands.",
            "Build a React table with virtual scrolling, sticky headers, and pinned columns.",
            "Create a React form that dynamically adds and removes fields based on user selections.",
            "Write a React component that renders Markdown with syntax-highlighted code blocks.",
            "Implement a React dashboard with resizable and draggable widget panels.",
            "Build a React breadcrumb navigation component generated from the current route.",
            "Create a React carousel with auto-play, dot indicators, and touch swipe support.",
            "Write a React component displaying paginated search results with highlighted query terms.",
            "Implement a React number input with increment/decrement buttons, min/max, and step support.",
            "Build a React multi-select dropdown with tag display, search filter, and create-new-option.",
            "Create a React color picker with hex input, RGB sliders, and saved color swatches.",
            "Write a React component that animates a counter from zero to a target value on scroll into view.",
            "Implement a React context menu on right-click with configurable menu items.",
            "Build a React floating action button that expands into multiple options with animation.",
            "Write a React table with inline row editing, validation, and per-row save/cancel controls.",
            "Create a React hook for managing complex nested form state with array fields.",
            "Implement a React sticky header that shrinks and becomes opaque as the user scrolls.",
            # Node.js
            "Write a Node.js Express REST API endpoint with cursor-based pagination for large result sets.",
            "Create a Node.js rate-limiting middleware using a sliding window algorithm keyed by IP address.",
            "Implement a Node.js function that streams a large file to the client using piped read streams.",
            "Build a Node.js job queue processor using Bull with retries, concurrency, and dead-letter queues.",
            "Write a Node.js JWT authentication middleware with role-based route authorization.",
            "Create a Node.js WebSocket server that broadcasts real-time updates to subscribed clients.",
            "Implement a Node.js CLI tool using Commander.js that processes CSV files and outputs JSON.",
            "Build a Node.js function to batch-upload files to AWS S3 with retries and progress reporting.",
            "Write a Node.js Express server with request logging, error middleware, and graceful shutdown.",
            "Create a Node.js cron job that sends daily email summaries using Nodemailer.",
            "Implement a Node.js GraphQL server using Apollo exposing a user and posts data model.",
            "Build a Node.js function that resizes and optimizes images using Sharp with caching.",
            "Write a Node.js Express API with request body validation using JSON Schema.",
            "Create a Node.js Redis caching layer for an Express API with TTL and cache invalidation.",
            "Implement a Node.js function that generates PDF reports from HTML templates using Puppeteer.",
            "Write a Node.js Server-Sent Events endpoint that streams live price updates.",
            "Create a Node.js function that implements OAuth 2.0 authorization code flow for GitHub login.",
            "Build a Node.js distributed lock using Redis SETNX for coordinating concurrent workers.",
            "Write a Node.js Express proxy that routes requests to backend services based on path prefix.",
            "Implement a Node.js circuit breaker pattern for resilient external API calls.",
            "Create a Node.js function that monitors a directory for new files and processes them automatically.",
            "Build a Node.js worker thread pool for CPU-intensive image processing tasks.",
            "Write a Node.js function that compresses HTTP responses with Brotli using zlib streams.",
            "Create a Node.js function to export a large MongoDB collection to CSV using streaming aggregation.",
            "Implement a Node.js health-check endpoint that verifies database, cache, and external service status.",
            "Write a Node.js function that generates and validates TOTP codes for two-factor authentication.",
            "Build a Node.js Express API that supports multipart file uploads with size and type limits.",
            "Create a Node.js function that implements exponential backoff retry for HTTP requests.",
            "Write a Node.js function that indexes documents in Elasticsearch and performs full-text search.",
            "Implement a Node.js pub/sub system using Redis channels for real-time messaging.",
            # TypeScript and Vanilla JS
            "Convert a JavaScript utility module to TypeScript adding interfaces, generics, and strict null checks.",
            "Write a TypeScript utility type that makes all nested properties of an object optional recursively.",
            "Implement a TypeScript generic function that deep-merges two objects preserving all nested keys.",
            "Create a TypeScript class with a fluent builder pattern for constructing configuration objects.",
            "Write a TypeScript decorator that validates method arguments using runtime type checks.",
            "Implement a TypeScript discriminated union to model API response states: loading, success, and error.",
            "Build a TypeScript function that deep-clones an object preserving Date instances and class prototypes.",
            "Write a TypeScript event bus module with typed event handlers and automatic memory leak prevention.",
            "Create a TypeScript utility for building URL query strings from strongly-typed parameter objects.",
            "Implement a TypeScript class providing a fluent API for constructing SQL query strings.",
            "Build a TypeScript LRU cache with configurable max size and optional TTL per entry.",
            "Write a TypeScript generic Stack data structure with push, pop, peek, and isEmpty operations.",
            "Create a TypeScript function that validates JSON at runtime using type guards.",
            "Implement a TypeScript pipe function composing multiple transformations with type inference.",
            "Write a TypeScript finite state machine for a checkout flow with validated transitions.",
            "Create a TypeScript function that parses and validates environment variables with typed defaults.",
            "Implement a JavaScript generator that lazily produces paginated API results.",
            "Write a JavaScript function that deep-freezes a nested object making all properties immutable.",
            "Create a JavaScript Proxy that intercepts property access and logs all reads and writes.",
            "Implement a JavaScript observer pattern with typed events and automatic unsubscribe support.",
            "Write a JavaScript function that implements a binary heap priority queue.",
            "Create a JavaScript function that parses CSV with configurable delimiter and quoting rules.",
            "Implement a JavaScript function that computes Levenshtein distance for fuzzy string matching.",
            "Write a JavaScript class wrapping WebSocket with auto-reconnect and message queuing.",
            "Create a JavaScript function that transforms a flat node list into a nested tree structure.",
            "Implement a JavaScript function that computes topological sort for a dependency graph.",
            "Write a JavaScript function that implements retry with exponential backoff for a Promise.",
            "Create a JavaScript function that batches multiple API calls using a DataLoader pattern.",
            "Implement a JavaScript function that creates an observable from a DOM event with cleanup.",
            "Write a service worker implementing stale-while-revalidate caching for API responses.",
            # Frameworks and tooling
            "Build a webpack plugin that analyzes bundle size and warns when a chunk exceeds a threshold.",
            "Write a Vite plugin that injects environment-specific configuration at build time.",
            "Create a Jest custom matcher for asserting on deeply nested object structures.",
            "Implement a Cypress end-to-end test for a multi-step checkout flow with API stubbing.",
            "Write an ESLint custom rule enforcing consistent error handling in async functions.",
            "Create a next.js middleware that redirects users based on A/B test bucket assignment.",
            "Implement a next.js API route with chunked file upload and server-side assembly.",
            "Write a next.js page using Incremental Static Regeneration with on-demand revalidation.",
            "Create a Svelte store with derived computed values and persistent localStorage sync.",
            "Write a Vue 3 composable for managing async state with cancellation on component unmount.",
            "Implement a NestJS interceptor that transforms response envelopes and logs request timing.",
            "Create a tRPC router with type-safe API procedures for a user management module.",
            "Write a Fastify plugin implementing request validation using JSON Schema.",
            "Implement a Remix loader that authenticates users and fetches session-scoped data.",
            "Create an Electron IPC handler that securely exposes file system operations to the renderer.",
            "Write a SolidJS signal-based store for managing a shopping cart with derived totals.",
            "Implement a GraphQL subscription resolver for real-time notification delivery.",
            "Create a Prisma schema and migration for a multi-tenant app with row-level security.",
            "Write a custom GitHub Actions workflow that runs Playwright tests with parallelization.",
            "Implement a Docker multi-stage build for a Node.js production image with minimal footprint.",
            # Algorithms and data structures
            "Write a JavaScript function implementing run-length encoding and decoding for strings.",
            "Create a JavaScript function that validates credit card numbers using the Luhn algorithm.",
            "Implement a JavaScript function computing the convex hull of a set of 2D points.",
            "Write a JavaScript function that implements a bloom filter for approximate set membership.",
            "Create a JavaScript function implementing LZ77-based string compression.",
            "Implement a JavaScript function computing topological sort for a dependency graph using DFS.",
            "Write a JavaScript function implementing a trie data structure for prefix-based autocomplete.",
            "Create a JavaScript function that computes a diff between two arrays using LCS.",
            "Implement a JavaScript doubly-linked list with O(1) insert and delete by reference.",
            "Write a JavaScript function that serializes and deserializes binary data using DataView.",
            # Additional React tasks
            "Create a React component that fetches and displays paginated GitHub repositories with search.",
            "Write a React hook that synchronizes component state with a URL query parameter.",
            "Implement a React form with Zod schema validation and react-hook-form integration.",
            "Build a React data table with column visibility toggles and persistent column order.",
            "Create a React component for a multi-level nested comment thread with collapse support.",
            "Write a React hook for managing keyboard shortcuts with conflict detection.",
            "Implement a React component that renders a gantt chart using CSS grid.",
            "Build a React virtualized tree view that renders only visible nodes for large datasets.",
            "Create a React component with ResizeObserver to adapt layout to container dimensions.",
            "Write a React hook that tracks and restores scroll position on navigation.",
            # Additional Node.js tasks
            "Write a Node.js streaming CSV parser that validates and transforms rows on the fly.",
            "Create a Node.js function using the cluster module to utilize all CPU cores for an HTTP server.",
            "Implement a Node.js middleware that extracts and validates request signatures for webhook security.",
            "Build a Node.js function that performs database connection pool management with health checks.",
            "Write a Node.js function using readline to process a large log file line by line.",
            "Create a Node.js Express server with CORS configuration supporting multiple allowed origins.",
            "Implement a Node.js task queue with priority levels and configurable concurrency limits.",
            "Build a Node.js function that computes a streaming SHA256 hash of a file without loading it.",
            "Write a Node.js function that reads a directory tree recursively and outputs a JSON structure.",
            "Create a Node.js function that implements basic HTTP digest authentication.",
            # Additional TypeScript tasks
            "Write a TypeScript template literal type that converts a dot-separated path to nested object access.",
            "Implement a TypeScript function that extracts the return type of a Promise-returning function.",
            "Create a TypeScript Higher-Kinded Type simulation for a functor interface.",
            "Write a TypeScript type that infers the argument types of a given function signature.",
            "Implement a TypeScript class that uses the Iterable protocol for custom for-of support.",
            "Create a TypeScript decorator factory that measures and logs method execution time.",
            "Write a TypeScript function that implements structural type checking with recursive constraints.",
            "Implement a TypeScript nominal type using a branded type pattern to prevent value confusion.",
            "Create a TypeScript utility type that converts all Promise-returning methods to synchronous.",
            "Write a TypeScript generic class implementing a typed event emitter with once and off support.",
            # Testing and tooling
            "Write a Playwright visual regression test that detects pixel-level UI changes between builds.",
            "Create a k6 load test script for a login endpoint measuring P95 response time under 100 VUs.",
            "Implement a Jest snapshot test for a React component that renders conditionally.",
            "Write a Cypress custom command that logs in via API and injects the session cookie.",
            "Create an MSW (Mock Service Worker) handler set for a complete user management API.",
            "Write a Vitest unit test suite for a pure utility module with 100% branch coverage.",
            "Implement a Storybook story for a complex form component with all validation states.",
            "Create a GitHub Actions matrix job that runs tests on Node 18, 20, and 22.",
            "Write a Renovate configuration to auto-merge patch updates but require review for major versions.",
            "Implement a custom ESLint rule that forbids importing from a deprecated internal module.",
            # Advanced patterns
            "Write a JavaScript function implementing a reactive state container like a minimal MobX.",
            "Create a JavaScript module that implements structural sharing for immutable data updates.",
            "Implement a JavaScript function for composing middleware chains like Express use().",
            "Write a JavaScript function that implements a transducer for efficient array processing.",
            "Create a JavaScript scheduler using requestIdleCallback for deferring non-critical work.",
            "Implement a JavaScript function that segments a large computation into time-sliced chunks.",
            "Write a JavaScript function using WeakRef and FinalizationRegistry for observable object lifetime.",
            "Create a JavaScript function that implements schema-based object coercion and defaults.",
            "Implement a JavaScript function that computes a stable hash of an arbitrary JSON value.",
            "Write a JavaScript REPL evaluator that sandboxes code using a Proxy-based scope.",
            # Performance and optimization
            "Write a JavaScript function that memoizes expensive renders using a reference-equality cache.",
            "Create a JavaScript module that pools and recycles objects to reduce GC pressure.",
            "Implement a JavaScript function that batches DOM mutations using requestAnimationFrame.",
            "Write a JavaScript function that uses SharedArrayBuffer for zero-copy data sharing with workers.",
            "Create a JavaScript function that streams large JSON parsing without full deserialization.",
            "Implement a JavaScript function that uses an IntersectionObserver to defer off-screen work.",
            "Write a JavaScript function that implements adaptive throttling based on frame rate.",
            "Create a JavaScript function that uses OffscreenCanvas for background image processing.",
            "Implement a JavaScript function that uses WASM for CPU-intensive numeric computation.",
            "Write a JavaScript function that profiles and visualizes event loop lag over time.",
            "Implement a JavaScript reactive form validation system using Proxy objects.",
            "Create a JavaScript function that streams Server-Sent Events and reconnects automatically.",
            "Write a JavaScript function that computes a Merkle tree root from a list of transactions.",
            "Implement a JavaScript rate limiter using a sliding window with a Redis-like in-memory store.",
            "Create a JavaScript function that encodes and decodes base64url strings for JWT headers.",
            "Write a JavaScript function that deep-merges two configuration objects with array concat support.",
            "Implement a JavaScript observable state container with middleware support like Redux.",
            "Create a JavaScript function that implements a persistent segment tree for historical queries.",
            "Write a JavaScript function that computes edit distance between two strings with backtracking.",
            "Implement a JavaScript async queue that processes tasks with configurable concurrency.",
        ],
    },
    "cpp": {
        "topic": "cpp",
        "followup_pool": ADVERSARIAL_FOLLOWUP_POOL["cpp"],
        "root_tasks": [
            # Memory management
            "Implement a custom memory pool allocator in C++ that pre-allocates a fixed block and serves fixed-size allocations.",
            "Write a RAII wrapper for a C-style file handle that ensures the file is always closed.",
            "Implement a reference-counted smart pointer class similar to std::shared_ptr from scratch.",
            "Create a unique_ptr-like class with custom deleter support and move semantics.",
            "Write a C++ function that detects and reports memory leaks in a small object pool.",
            "Implement a stack allocator that allocates from a fixed buffer and supports rollback.",
            "Create a C++ intrusive linked list where nodes embed the list pointers directly.",
            "Write a C++ function that safely copies overlapping memory regions without undefined behavior.",
            "Implement a buddy allocator that splits and merges power-of-two blocks for allocation.",
            "Create a C++ scope guard that executes a lambda on scope exit regardless of exceptions.",
            # Templates and generics
            "Write a C++ variadic template function that computes the sum of any number of numeric arguments.",
            "Implement a type-erased callable class similar to std::function without heap allocation for small callables.",
            "Create a C++ template metaprogramming function that computes Fibonacci numbers at compile time.",
            "Write a C++ CRTP base class implementing the Curiously Recurring Template Pattern for static polymorphism.",
            "Implement a C++ type list and a compile-time index lookup using template specialization.",
            "Create a C++ tuple implementation supporting get<N> and structured bindings.",
            "Write a C++ concept (C++20) that constrains a template parameter to sortable containers.",
            "Implement a C++ template function that applies a transformation to each element of a heterogeneous tuple.",
            "Create a C++ expression template class for lazy evaluation of vector arithmetic.",
            "Write a C++ policy-based class using template parameters to inject sorting and comparison strategies.",
            # STL and algorithms
            "Implement a custom STL-compatible forward iterator for a singly-linked list.",
            "Write a C++ algorithm that performs an in-place merge sort on a std::vector.",
            "Create a C++ comparator for std::sort that sorts structs by multiple fields with configurable order.",
            "Implement a C++ range adaptor using C++20 ranges that filters and transforms lazily.",
            "Write a C++ function using std::transform and std::back_inserter to build a mapped result vector.",
            "Implement a C++ function using std::accumulate with a custom binary operation for a running product.",
            "Create a C++ sliding window algorithm on a deque that computes a rolling maximum in O(n).",
            "Write a C++ function that uses std::partition to separate elements satisfying a predicate in place.",
            "Implement a C++ generic binary search that works on any sorted random-access range.",
            "Create a C++ function using std::nth_element to find the k-th smallest element in O(n) average time.",
            # Concurrency
            "Write a C++ thread pool that distributes tasks across worker threads using a lock-free queue.",
            "Implement a C++ producer-consumer queue using std::mutex and std::condition_variable.",
            "Create a C++ read-write lock allowing multiple concurrent readers but exclusive writers.",
            "Write a C++ atomic reference counter for a shared resource with acquire-release semantics.",
            "Implement a C++ barrier synchronization primitive using atomics for a fixed number of threads.",
            "Create a C++ future/promise pattern for returning results from background threads.",
            "Write a C++ lock-free stack using compare-and-swap (CAS) atomic operations.",
            "Implement a C++ thread-safe singleton using double-checked locking with std::call_once.",
            "Create a C++ task scheduler that executes tasks at specified time points using a priority queue.",
            "Write a C++ pipeline where multiple stages run concurrently connected by shared queues.",
            # Data structures
            "Implement a C++ AVL tree with insert, delete, and in-order traversal.",
            "Write a C++ hash map using open addressing with linear probing and automatic rehashing.",
            "Create a C++ red-black tree with insert and search operations.",
            "Implement a C++ skip list supporting O(log n) average-case insert and search.",
            "Write a C++ min-max heap that supports O(1) access to both minimum and maximum.",
            "Create a C++ segment tree supporting range sum queries and point updates.",
            "Implement a C++ Fenwick tree (Binary Indexed Tree) for prefix sum queries.",
            "Write a C++ disjoint set union (union-find) with path compression and union by rank.",
            "Create a C++ LRU cache using a combination of std::list and std::unordered_map.",
            "Implement a C++ trie for string storage with prefix search and autocomplete.",
            # Modern C++ and systems
            "Write a C++ coroutine (C++20) that lazily generates an infinite Fibonacci sequence.",
            "Implement a C++ span-based function that processes a subrange of a buffer without copying.",
            "Create a C++ structured binding decomposition for a custom aggregate type.",
            "Write a C++ std::variant visitor that handles all types in a type-safe union.",
            "Implement a C++ string_view-based parser that avoids allocations while tokenizing a CSV line.",
            "Create a C++ lambda that captures by move for use in a std::thread constructor.",
            "Write a C++ function that serializes a struct to binary using placement new and std::byte.",
            "Implement a C++ function that measures cache line effects by comparing sequential vs. strided access.",
            "Create a C++ SIMD-accelerated function that computes the dot product of two float arrays using intrinsics.",
            "Write a C++ function that uses __builtin_expect to hint the branch predictor in a hot loop.",
            # OOP and design patterns
            "Implement the Command design pattern in C++ for an undo/redo text editor.",
            "Write a C++ Observer pattern implementation with weak reference subscribers to avoid memory leaks.",
            "Create a C++ Factory Method for creating different shape objects from a string type name.",
            "Implement a C++ Strategy pattern for interchangeable sorting algorithms on a dataset.",
            "Write a C++ Decorator pattern that adds logging and caching to an existing interface.",
            "Create a C++ Flyweight pattern for efficiently sharing immutable string data.",
            "Implement a C++ Chain of Responsibility for processing HTTP-like request headers.",
            "Write a C++ State machine for a TCP connection lifecycle using the State pattern.",
            "Create a C++ Proxy pattern wrapping a remote resource with lazy initialization.",
            "Implement a C++ Template Method pattern for a configurable data processing pipeline.",
            # Algorithms and math
            "Write a C++ function that implements Dijkstra's shortest path algorithm on an adjacency list.",
            "Implement a C++ function computing the longest common subsequence of two strings.",
            "Create a C++ dynamic programming solution for the 0/1 knapsack problem.",
            "Write a C++ function using bit manipulation to count set bits in a 64-bit integer.",
            "Implement a C++ fast Fourier transform (FFT) for polynomial multiplication.",
            "Create a C++ function implementing the KMP string search algorithm.",
            "Write a C++ function solving the N-Queens problem using backtracking.",
            "Implement a C++ function computing the edit distance between two strings.",
            "Create a C++ function implementing Huffman encoding for text compression.",
            "Write a C++ function computing a topological sort of a directed acyclic graph.",
            # File I/O and networking
            "Write a C++ function that reads a large binary file in chunks and parses a custom record format.",
            "Implement a C++ CSV parser that handles quoted fields, escaped characters, and multi-line values.",
            "Create a C++ function that serializes a vector of structs to a memory-mapped file.",
            "Write a C++ TCP server using POSIX sockets that handles multiple clients with select().",
            "Implement a C++ HTTP/1.1 response parser that handles chunked transfer encoding.",
            "Create a C++ function that monitors a directory for changes using inotify on Linux.",
            "Write a C++ function that atomically writes a file using a temp file and rename.",
            "Implement a C++ function that reads configuration from an INI file into a std::map.",
            "Create a C++ logging class that writes asynchronously to a file with log rotation.",
            "Write a C++ function that implements a simple LZ4-style block compression.",
            # Testing and debugging
            "Write a C++ Google Test unit test suite for a binary search tree implementation.",
            "Create a C++ mock object for a network interface using a virtual function interface.",
            "Implement a C++ sanitizer-friendly memory pool that catches use-after-free.",
            "Write a C++ function with AddressSanitizer annotations to detect buffer overflows.",
            "Create a C++ benchmark using Google Benchmark comparing two sort implementations.",
            "Implement a C++ fuzzing harness for a custom string parser using libFuzzer.",
            "Write a C++ function with static_assert checks to validate template argument constraints.",
            "Create a C++ exception-safe function with strong exception guarantee using copy-and-swap.",
            "Implement a C++ RAII lock guard that detects and throws on re-entrant acquisition.",
            "Write a C++ utility that tracks object construction and destruction counts for leak detection.",
            # Build and packaging
            "Write a CMakeLists.txt that builds a C++ library with public and private include directories.",
            "Create a CMake target_compile_options configuration for strict warning flags across compilers.",
            "Implement a Conan recipe for packaging and distributing a C++ library.",
            "Write a CMake ExternalProject_Add integration for a third-party dependency.",
            "Create a CMake install configuration that generates package config files for downstream projects.",
            "Write a C++ cross-compilation CMake toolchain file for an ARM Linux target.",
            "Implement a CMake custom command that auto-generates C++ header files from a spec file.",
            "Create a vcpkg manifest file for managing C++ dependencies in a project.",
            "Write a C++ code coverage configuration using gcov and lcov with a CMake target.",
            "Implement a C++ static analysis configuration using Clang-Tidy with CMake integration.",
            # Additional memory and RAII
            "Write a C++ arena allocator that bumps a pointer and supports batch deallocation.",
            "Implement a C++ weak_ptr-based cache that evicts entries when no owner holds a shared_ptr.",
            "Create a C++ move-only handle type wrapping an OS resource with move constructor and operator=.",
            "Write a C++ function that allocates and frees aligned memory for SIMD operations.",
            "Implement a C++ small buffer optimization for a type-erased storage class.",
            "Create a C++ pool allocator specialized for a fixed-size node type used in a tree.",
            "Write a C++ class with pimpl idiom to hide private members from the header.",
            "Implement a C++ string interning table using std::unordered_set for deduplication.",
            "Create a C++ scope guard that executes a cleanup lambda on scope exit unconditionally.",
            "Write a C++ reference wrapper class that prevents copying but allows rebinding.",
            # Additional concurrency
            "Write a C++ work-stealing deque for a multi-threaded task scheduler.",
            "Implement a C++ semaphore using std::counting_semaphore for resource rate limiting.",
            "Create a C++ message-passing actor with a typed mailbox using std::queue and std::mutex.",
            "Write a C++ lock-free multiple-producer multiple-consumer bounded ring buffer.",
            "Implement a C++ thread-local cache using thread_local storage for per-thread memoization.",
            "Create a C++ pipeline where multiple stages run concurrently connected by bounded queues.",
            "Write a C++ function using OpenMP to parallelize a matrix-vector multiplication.",
            "Implement a C++ async file read using POSIX aio and a completion callback.",
            "Create a C++ condition variable-based event loop that processes a task queue.",
            "Write a C++ atomic flag-based spin lock with exponential backoff.",
            # Additional templates and metaprogramming
            "Write a C++ type traits class that detects whether a type has a serialize member function.",
            "Implement a C++ constexpr sorted array map for compile-time key-value lookup.",
            "Create a C++ fold expression that concatenates a variadic pack of std::string_view.",
            "Write a C++ mixin chain that adds serialization and comparison step by step via inheritance.",
            "Implement a C++ tag dispatch pattern to select an algorithm based on iterator category.",
            "Create a C++ type-safe units library where meters and kilograms cannot be accidentally added.",
            "Write a C++ SFINAE function that falls back to a default for non-arithmetic types.",
            "Implement a C++ generator coroutine (C++20) that lazily yields Fibonacci numbers.",
            "Create a C++ partial specialization for a matrix class optimized for 2x2 and 3x3 sizes.",
            "Write a C++ consteval function that validates a format string at compile time.",
            # Additional STL and algorithms
            "Write a C++ function using std::ranges::sort with a projection to sort by a struct member.",
            "Implement a C++ circular buffer using a fixed-size array with head and tail indices.",
            "Create a C++ function using std::views::zip to iterate two ranges in synchronized lockstep.",
            "Write a C++ function using std::stable_partition to reorder elements in place by predicate.",
            "Implement a C++ function using std::lower_bound with a custom comparator for interval overlap.",
            "Create a C++ bidirectional iterator adapter for a custom ring-buffer container.",
            "Write a C++ function using std::transform_reduce for a parallel inner product.",
            "Implement a C++ function using std::adjacent_find to detect the first consecutive duplicate.",
            "Create a C++ function using std::inplace_merge to combine two sorted halves in place.",
            "Write a C++ function using std::nth_element to efficiently partition into top-k and rest.",
            # Additional algorithms and math
            "Write a C++ function implementing the Rabin-Karp rolling hash for multi-pattern substring search.",
            "Implement a C++ function computing modular exponentiation using fast power algorithm.",
            "Create a C++ function solving a system of linear equations using Gaussian elimination with pivoting.",
            "Write a C++ function implementing a B-tree node with insert, split, and search operations.",
            "Implement a C++ function for Aho-Corasick multi-pattern string search with failure links.",
            "Create a C++ function that computes the maximum bipartite matching using augmenting paths.",
            "Write a C++ function implementing the Miller-Rabin probabilistic primality test.",
            "Implement a C++ function that solves the 0/1 knapsack problem with memoization.",
            "Create a C++ function computing the longest palindromic substring using Manacher's algorithm.",
            "Write a C++ function implementing a segment tree with lazy propagation for range updates.",
            # Additional networking and I/O
            "Write a C++ async TCP server using non-blocking sockets and an epoll event loop.",
            "Implement a C++ HTTP/1.1 request parser using a state machine with chunked transfer support.",
            "Create a C++ function that sends a UDP datagram and waits for an ACK with configurable timeout.",
            "Write a C++ function implementing a simple TLV binary protocol encoder and decoder.",
            "Implement a C++ function for zero-copy file serving using sendfile on Linux.",
            "Create a C++ file watcher using inotify that triggers callbacks on file modification.",
            "Write a C++ function that computes a rolling Adler-32 checksum of a data stream.",
            "Implement a C++ function that serializes a struct to JSON using recursive reflection.",
            "Create a C++ function that reads and writes big-endian integers portably across architectures.",
            "Write a C++ function using mmap for zero-copy reading of a large binary file.",
            # Additional OOP patterns
            "Implement the Null Object pattern in C++ to eliminate null pointer checks in a logging hierarchy.",
            "Write a C++ Abstract Factory for creating platform-specific UI widget families.",
            "Create a C++ Memento pattern for save and restore of a text document editor state.",
            "Implement a C++ Bridge pattern separating a shape abstraction from its OpenGL rendering.",
            "Write a C++ Specification pattern for composable and reusable business rule predicates.",
            "Create a C++ Plugin Manager that loads shared libraries and discovers factory symbols at runtime.",
            "Implement a C++ Reactive Property that notifies observers only when its value actually changes.",
            "Write a C++ Object Pool that reuses pre-allocated objects and tracks checkout and return.",
            "Create a C++ Event Bus with compile-time event IDs to avoid runtime type erasure overhead.",
            "Implement a C++ Visitor pattern for computing metrics on an expression tree without modifying nodes.",
            "Write a C++ lock-free stack using compare-and-swap to push and pop without a mutex.",
            "Create a C++ function that implements a ring buffer with power-of-two size for cache efficiency.",
            "Implement a C++ intrusive doubly-linked list where nodes carry the prev and next pointers.",
            "Write a C++ function using CRTP to implement a static interface for serializable objects.",
            "Create a C++ concept (C++20) that validates a type is a range with a value_type of int.",
            "Implement a C++ function that computes matrix multiplication using Strassen's algorithm.",
            "Write a C++ function that implements radix sort for 32-bit unsigned integers.",
            "Create a C++ function that detects overlapping intervals and merges them into a minimal set.",
            "Implement a C++ function that builds a suffix array from a string in O(n log n).",
            "Write a C++ function that computes the convex hull of a point set using the Graham scan algorithm.",
            "Create a C++ function that implements a simple arena-based JSON parser.",
            "Implement a C++ function that computes the longest common substring of two strings using DP.",
            "Write a C++ function that implements a Van Emde Boas tree for integer sets.",
            "Create a C++ function that computes a perfect hash function for a static known key set.",
            "Implement a C++ function that implements the Z-algorithm for linear-time pattern matching.",
            "Write a C++ function that implements an order-statistic tree using an augmented BST.",
            "Create a C++ function that implements persistent data structures using path copying.",
            "Implement a C++ function that solves the N-Queens problem using bitmask pruning.",
            "Write a C++ function that implements a cache-oblivious matrix transpose.",
            "Create a C++ function that implements a lock-free read-copy-update (RCU) mechanism.",
        ],
    },
    "sql": {
        "topic": "sql",
        "followup_pool": ADVERSARIAL_FOLLOWUP_POOL["sql"],
        "root_tasks": [
            # Basic queries
            "Write a SQL query to find all customers who made more than 3 purchases in the last 30 days.",
            "Create a SQL query to retrieve the top 10 products by revenue for each category.",
            "Write a SQL query to find duplicate email addresses in a users table with their occurrence count.",
            "Create a SQL query to calculate the month-over-month revenue growth percentage.",
            "Write a SQL query to find all orders that contain at least one product from a specific supplier.",
            "Create a SQL query to retrieve users who registered but never placed an order.",
            "Write a SQL query to calculate the running total of sales ordered by date.",
            "Create a SQL query to find the second highest salary in each department.",
            "Write a SQL query to retrieve all employees who report directly or indirectly to a given manager.",
            "Create a SQL query to find products that have never been ordered.",
            # Joins
            "Write a SQL query joining orders, customers, and products to produce a full invoice detail report.",
            "Create a SQL query using a self-join to find pairs of employees in the same department.",
            "Write a SQL query using a cross join to generate a complete schedule grid for all time slots and rooms.",
            "Create a SQL query using a non-equi join to assign commission tiers based on sale amount ranges.",
            "Write a SQL query to compare current period sales with the same period last year using a self-join.",
            "Create a SQL query using multiple LEFT JOINs to enrich a fact table with dimension attributes.",
            "Write a SQL query that finds all products with no matching inventory record using an anti-join.",
            "Create a SQL query using FULL OUTER JOIN to reconcile two account lists and flag discrepancies.",
            "Write a SQL query joining three tables to compute a weighted average score per student.",
            "Create a SQL query using a lateral join to fetch the most recent 3 orders per customer.",
            # Aggregations
            "Write a SQL query to compute the median order value per product category.",
            "Create a SQL query to calculate the 90th percentile response time per API endpoint.",
            "Write a SQL query to produce a frequency distribution histogram of product prices in 10 buckets.",
            "Create a SQL query to compute the Gini coefficient of income distribution across regions.",
            "Write a SQL query to aggregate customer transactions into daily summaries with open and close balances.",
            "Create a SQL query to compute the customer churn rate by cohort month.",
            "Write a SQL query to calculate the average time between consecutive orders per customer.",
            "Create a SQL query to find the most common sequence of page views in a session.",
            "Write a SQL query to compute the conversion funnel drop-off rates between each step.",
            "Create a SQL query to calculate the net promoter score from a ratings table.",
            # Window functions
            "Write a SQL query using ROW_NUMBER to deduplicate rows keeping only the latest record per key.",
            "Create a SQL query using RANK to assign medals (1st, 2nd, 3rd) per category by score.",
            "Write a SQL query using LEAD and LAG to compute the time gap between consecutive events per user.",
            "Create a SQL query using FIRST_VALUE and LAST_VALUE to compute session start and end values.",
            "Write a SQL query using a sliding window frame to compute a 7-day moving average of daily sales.",
            "Create a SQL query using NTILE to divide customers into deciles by lifetime value.",
            "Write a SQL query using SUM OVER PARTITION to compute the percentage contribution of each row.",
            "Create a SQL query using window functions to identify streak lengths of consecutive wins.",
            "Write a SQL query using DENSE_RANK to compute percentile ranks within each group.",
            "Create a SQL query using window functions to detect gaps in a sequence of events.",
            # CTEs and subqueries
            "Write a recursive CTE to traverse an employee hierarchy and compute the depth of each node.",
            "Create a CTE chain that builds a customer segment summary in four steps.",
            "Write a SQL query using a correlated subquery to find products priced above their category average.",
            "Create a CTE to compute cohort sizes and join it with retention data for a retention matrix.",
            "Write a SQL query using a lateral subquery to compute the top-3 tags per article.",
            "Create a recursive CTE to generate a date series between two given dates.",
            "Write a SQL query using nested subqueries to compute Z-scores for each transaction.",
            "Create a CTE that identifies sessions from a raw event log based on 30-minute inactivity gaps.",
            "Write a SQL query using an EXISTS subquery to find customers with at least two orders over $100.",
            "Create a CTE-based query to compute a 28-day rolling retention rate per acquisition channel.",
            # Stored procedures
            "Write a stored procedure to upsert customer records and log changes to an audit table.",
            "Create a stored procedure that processes a batch of pending orders and updates inventory.",
            "Write a stored procedure to compute and store daily summary metrics with error handling and logging.",
            "Create a stored procedure that validates data quality rules and raises exceptions on violations.",
            "Write a stored procedure to archive records older than N days to a history table and delete originals.",
            "Create a stored procedure that generates a report and inserts it into a report_cache table.",
            "Write a stored procedure with cursor-based processing to apply a complex transformation row by row.",
            "Create a stored procedure that implements a retry loop for a flaky external procedure call.",
            "Write a stored procedure to rebuild all fragmented indexes in a database schema.",
            "Create a parameterized stored procedure that supports optional filters using dynamic SQL safely.",
            # Indexes and optimization
            "Write a SQL script to create a composite index for a slow query identified by EXPLAIN ANALYZE.",
            "Create a partial index on an orders table covering only unfulfilled orders.",
            "Write a SQL script to identify and remove redundant indexes in a schema.",
            "Create a covering index that eliminates a costly key lookup in a frequently-run query.",
            "Write a SQL query that avoids a function call on an indexed column that prevents index use.",
            "Create an index for a full-text search query and rewrite the query to use it.",
            "Write a SQL script to monitor index usage statistics and flag unused indexes.",
            "Create a filtered index to accelerate queries on a soft-deleted records pattern.",
            "Write a SQL script to update statistics and rebuild stale indexes for a specific table.",
            "Create a spatial index and rewrite a proximity search query to use it.",
            # Transactions and concurrency
            "Write a SQL transaction that transfers funds between two accounts with ACID guarantees.",
            "Create a SQL script that implements optimistic concurrency control using a version column.",
            "Write a SQL transaction that inserts a parent and child record atomically.",
            "Create a SQL deadlock scenario and rewrite the queries with consistent lock ordering to avoid it.",
            "Write a SQL script that uses SAVEPOINT to implement nested transaction rollback.",
            "Create a SQL trigger that enforces a business rule and logs violations to an audit table.",
            "Write a SQL script implementing row-level locking with SELECT FOR UPDATE SKIP LOCKED for a job queue.",
            "Create a SQL transaction that uses serializable isolation to prevent phantom reads.",
            "Write a SQL script to detect and kill long-running queries blocking other sessions.",
            "Create a SQL function that generates a UUID and uses it as a default value for a primary key.",
            # DDL and schema design
            "Write a SQL script to add a NOT NULL column to a large live table without a full table lock.",
            "Create a SQL schema for a multi-tenant SaaS application with row-level tenant isolation.",
            "Write a SQL script to implement table partitioning by month for a time series events table.",
            "Create a SQL schema for a polymorphic association using a discriminator column.",
            "Write a SQL script to rename a column in a production database with zero downtime.",
            "Create a SQL view that abstracts a complex join into a simple queryable interface.",
            "Write a SQL materialized view with a refresh strategy for a dashboard summary table.",
            "Create a SQL schema with proper foreign key constraints and cascading delete rules.",
            "Write a SQL script to migrate a varchar primary key to an integer primary key safely.",
            "Create a SQL check constraint enforcing a date range and a lookup value simultaneously.",
            # Administration and monitoring
            "Write a SQL script to compute table bloat and identify tables needing VACUUM FULL.",
            "Create a SQL query to find the top-10 slowest queries from the pg_stat_statements view.",
            "Write a SQL script to generate a data dictionary from information_schema.",
            "Create a SQL query that computes cache hit rates for tables and indexes.",
            "Write a SQL script to clone a production schema into a staging environment.",
            "Create a SQL query monitoring long-running transactions and their held locks.",
            "Write a SQL script to rotate a partition table by dropping the oldest partition and adding a new one.",
            "Create a SQL query showing replication lag between primary and read replica.",
            "Write a SQL script to generate test data using generate_series for load testing.",
            "Create a SQL script that computes dependency order for tables to safely truncate a schema.",
            # Additional window functions
            "Write a SQL query using window functions to compute a 30-day trailing average for each product.",
            "Create a SQL query using LISTAGG or STRING_AGG to concatenate ordered values per group.",
            "Write a SQL query using PERCENT_RANK to assign a relative rank within each department.",
            "Create a SQL query using CUME_DIST to find the cumulative distribution of salaries.",
            "Write a SQL query using window frame ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW for a running count.",
            "Create a SQL query using LAG to compute day-over-day change and flag reversals.",
            "Write a SQL query using RATIO_TO_REPORT (or equivalent) to compute each row's share of a group total.",
            "Create a SQL query using window functions to gap-fill missing dates in a sparse time series.",
            "Write a SQL query using FIRST_VALUE IGNORE NULLS to carry forward the last non-null value.",
            "Create a SQL query using window functions to identify the longest streak of daily logins.",
            # Additional CTEs and advanced queries
            "Write a recursive CTE to find all ancestors of a given node in a closure table.",
            "Create a SQL query using a CTE to compute the shortest path in a weighted graph using Dijkstra.",
            "Write a SQL query using PIVOT to transpose rows into columns for a report.",
            "Create a SQL query using UNPIVOT to normalize a wide table into a key-value long format.",
            "Write a SQL query using lateral joins to compute per-row top-N without window functions.",
            "Create a SQL query that uses a CTE to deduplicate based on a priority ranking per group.",
            "Write a SQL query using a recursive CTE to generate a Fibonacci sequence up to N terms.",
            "Create a SQL query using a CTE to compute a transition matrix from a sequence of states.",
            "Write a SQL query using multiple CTEs to build a customer 360 view joining 5 tables.",
            "Create a SQL query using a CTE to identify orphaned records with no parent reference.",
            # Additional optimization
            "Write a SQL query rewritten to use an index on a JSON column using a generated column.",
            "Create a SQL query that avoids a full table scan by rewriting an OR condition to UNION ALL.",
            "Write a SQL query that eliminates a sort operation by choosing an index with matching order.",
            "Create a SQL query rewritten from a correlated subquery to a JOIN for better performance.",
            "Write a SQL script to create a covering index eliminating a key lookup in a hot query.",
            "Create a SQL query using EXISTS instead of IN to improve performance on large subquery result sets.",
            "Write a SQL query that avoids implicit type conversion preventing index use on a date column.",
            "Create a SQL script to analyze query plan differences between two equivalent query formulations.",
            "Write a SQL query using hash joins hinted explicitly for a large join on unindexed columns.",
            "Create a SQL script to identify parameter sniffing issues and implement a fix using OPTION RECOMPILE.",
            # Additional stored procedures and functions
            "Write a SQL scalar function to compute the Haversine distance between two lat/lon coordinates.",
            "Create a SQL table-valued function that returns a calendar table for a given date range.",
            "Write a stored procedure that implements a merge (upsert) with conflict resolution strategy.",
            "Create a SQL function that parses a comma-separated string into a table of values.",
            "Write a stored procedure that runs a data quality check and populates an error log table.",
            "Create a SQL function that computes a moving average using a recursive CTE.",
            "Write a stored procedure to perform a soft-delete and cascade it to related tables.",
            "Create a SQL trigger that prevents deletion of a parent record with existing children.",
            "Write a stored procedure that generates and executes a dynamic pivot query.",
            "Create a SQL function that normalizes a phone number string to E.164 format.",
            # Additional schema and DDL
            "Write a SQL script to add a foreign key constraint to an existing table with data validation.",
            "Create a SQL schema for a hierarchical category system using the adjacency list model.",
            "Write a SQL script to split a large monolithic table into two normalized tables.",
            "Create a SQL schema for a billing system with invoices, line items, and payments.",
            "Write a SQL script implementing a slowly changing dimension type 2 with effective dates.",
            "Create a SQL schema for a multi-currency financial system with exchange rates.",
            "Write a SQL script to implement row versioning for an audit trail using temporal tables.",
            "Create a SQL schema for an event sourcing pattern with an events and snapshots table.",
            "Write a SQL script to implement table inheritance using a parent and child table pattern.",
            "Create a SQL schema for a tagging system with many-to-many entity-tag relationships.",
            # Additional analytics
            "Write a SQL query to compute a customer segmentation RFM (Recency, Frequency, Monetary) score.",
            "Create a SQL query to compute a product affinity matrix showing co-purchase frequencies.",
            "Write a SQL query to calculate the inventory turnover ratio per product per quarter.",
            "Create a SQL query to compute the average revenue per user (ARPU) by acquisition channel.",
            "Write a SQL query to detect anomalous transaction amounts using a z-score threshold.",
            "Create a SQL query to compute the day-1, day-7, and day-30 retention for each app version.",
            "Write a SQL query to calculate the payback period for a marketing campaign by cohort.",
            "Create a SQL query to identify the top-N customers by revenue contribution using Pareto analysis.",
            "Write a SQL query to compute a weighted NPS score from a ratings and weights table.",
            "Create a SQL query to forecast next month's revenue using a simple linear extrapolation on monthly data.",
            # Additional administration
            "Write a SQL script to identify tables with no primary key and report them.",
            "Create a SQL query to find the largest tables by row count and data size.",
            "Write a SQL script to transfer data between databases using INSERT INTO SELECT.",
            "Create a SQL query to identify missing indexes suggested by query optimizer statistics.",
            "Write a SQL script to check and repair integrity constraint violations in a dataset.",
            "Create a SQL script to generate all DROP TABLE statements for a schema in dependency order.",
            "Write a SQL script to compare row counts across environments to detect sync drift.",
            "Create a SQL query to find queries with high buffer cache miss rates from execution statistics.",
            "Write a SQL script to grant minimal permissions to a read-only reporting role.",
            "Create a SQL script to rotate sensitive data by re-encrypting with a new key.",
            "Write a SQL query using a window function to detect sudden spikes in metric values.",
            "Create a SQL query to compute a session funnel conversion rate per marketing campaign.",
            "Write a SQL query to calculate the average order value by customer segment and month.",
            "Create a SQL query to identify customers at risk of churn based on declining purchase frequency.",
            "Write a SQL query to compute the market basket association rules for pairs of products.",
            "Create a SQL script to extract, transform, and load daily data into a star schema fact table.",
            "Write a SQL query to find the first and last event for each user in a session.",
            "Create a SQL query to compute a moving 7-day sum of events per user.",
            "Write a SQL script to create a denormalized reporting table from a normalized schema.",
            "Create a SQL query to flag transactions outside business hours using time-of-day filtering.",
            "Write a SQL query to compute year-over-year growth rates for each product category.",
            "Create a SQL query to identify inactive accounts with no activity in the past 90 days.",
            "Write a SQL script to summarize data quality issues: nulls, duplicates, and out-of-range values.",
            "Create a SQL query to compute the median time between first and second purchase per customer.",
            "Write a SQL query to build a user journey table showing each step and time between steps.",
            "Create a SQL function that returns the fiscal quarter for a given date.",
            "Write a SQL query to compute the share of wallet (SOW) percentage per customer.",
            "Create a SQL query to find products that are frequently returned and flag their return rate.",
            "Write a SQL script to generate a monthly rollup table from daily granularity data.",
            "Create a SQL query using a lateral join to compute a running top-3 ranking per group.",
            "Write a SQL query to detect sequences where a metric crosses a threshold twice in a row.",
            "Create a SQL script to rebuild a corrupted sequence generator to fill missing IDs.",
            "Write a SQL query to compute the distribution of session lengths in seconds.",
            "Create a SQL query to summarize API error rates by endpoint and error code.",
            "Write a SQL query to generate a report of new vs. returning customers per month.",
            "Create a SQL query to find the most recent status change for each order.",
            "Write a SQL script to archive completed orders older than 2 years to a history table.",
            "Create a SQL query to compute average latency percentiles P50, P90, P95, P99.",
            "Write a SQL query to identify outlier stores with revenue deviating more than 2 sigma.",
            "Create a SQL query to compute a weighted average rating accounting for review count.",
        ],
    },
    "java": {
        "topic": "java",
        "followup_pool": ADVERSARIAL_FOLLOWUP_POOL["java"],
        "root_tasks": [
            # Spring Boot
            "Create a Spring Boot REST controller that exposes CRUD endpoints for a User entity with validation.",
            "Write a Spring Boot service that calls an external REST API using RestTemplate with retry logic.",
            "Implement a Spring Boot JWT authentication filter that validates tokens on every request.",
            "Create a Spring Boot scheduled task that runs a data cleanup job every night at 2 AM.",
            "Write a Spring Boot exception handler using @ControllerAdvice to return standardized error responses.",
            "Implement a Spring Boot WebSocket endpoint that sends real-time stock price updates.",
            "Create a Spring Boot batch job using Spring Batch to process CSV files and write to a database.",
            "Write a Spring Boot integration with Kafka to consume and publish domain events.",
            "Implement a Spring Boot configuration class that sets up a connection pool with HikariCP.",
            "Create a Spring Boot actuator custom endpoint that exposes application health metrics.",
            "Write a Spring Boot @Transactional service that orchestrates multi-step database operations.",
            "Implement a Spring Boot file upload endpoint that saves files to S3 and metadata to a database.",
            "Create a Spring Boot cache configuration using Redis with configurable TTL per cache region.",
            "Write a Spring Boot multi-tenant configuration that routes database connections by tenant ID.",
            "Implement a Spring Boot API versioning strategy using URI path versioning.",
            "Create a Spring Boot OpenAPI configuration that generates interactive API documentation.",
            "Write a Spring Boot rate limiting filter using Bucket4j with per-user limits.",
            "Implement a Spring Boot event listener that sends email notifications on domain events.",
            "Create a Spring Boot circuit breaker integration using Resilience4j.",
            "Write a Spring Boot request logging filter that records request/response with correlation IDs.",
            # Design patterns
            "Implement the Builder design pattern in Java for constructing complex SQL query objects.",
            "Write a Java Observer pattern for a stock price notification system with multiple subscribers.",
            "Create a Java Factory Method for instantiating payment processors from a payment type enum.",
            "Implement the Java Strategy pattern for interchangeable discount calculation algorithms.",
            "Write a Java Decorator pattern that adds caching and logging to a repository interface.",
            "Create a Java Chain of Responsibility for processing HTTP request filters.",
            "Implement a Java Command pattern for an undo/redo text editor operation history.",
            "Write a Java Composite pattern for a file system tree with files and directories.",
            "Create a Java Proxy pattern for a lazy-loading database entity.",
            "Implement the Java Template Method pattern for a configurable report generator.",
            "Write a Java Singleton implementation that is thread-safe and handles serialization.",
            "Create a Java Flyweight pattern for efficiently sharing font metadata across text runs.",
            "Implement a Java State machine for an order lifecycle with valid state transitions.",
            "Write a Java Visitor pattern for computing different metrics on an AST.",
            "Create a Java Mediator pattern for decoupled communication between UI components.",
            # Concurrency
            "Write a Java thread-safe counter using AtomicInteger and compare it to a synchronized version.",
            "Implement a Java producer-consumer queue using BlockingQueue and multiple threads.",
            "Create a Java CompletableFuture pipeline that fetches data from three APIs and merges results.",
            "Write a Java ReentrantReadWriteLock wrapper for a thread-safe in-memory cache.",
            "Implement a Java scheduled executor that runs tasks at fixed rate with exception handling.",
            "Create a Java fork-join task that computes a parallel merge sort on a large array.",
            "Write a Java CountDownLatch-based test that verifies concurrent operations complete correctly.",
            "Implement a Java ConcurrentHashMap-based frequency counter for a streaming word count.",
            "Create a Java StampedLock optimistic read pattern for a frequently-read, rarely-written cache.",
            "Write a Java virtual thread (Project Loom) based HTTP client for high-concurrency requests.",
            # Collections and Streams
            "Write a Java Stream pipeline that groups employees by department and computes average salary.",
            "Create a Java function using Collectors.toMap that merges duplicate keys using a merge function.",
            "Implement a Java Comparator chain that sorts a list by multiple fields with null-safe handling.",
            "Write a Java Stream that flattens a nested list of lists and removes duplicates.",
            "Create a Java function using Stream.reduce to implement a custom string join with prefix and suffix.",
            "Implement a Java collector that accumulates elements into a custom summary statistics object.",
            "Write a Java function using optional chaining to safely navigate a deep object graph.",
            "Create a Java stream pipeline that reads a CSV file lazily and processes it line by line.",
            "Implement a Java function using Map.computeIfAbsent for efficient memoized recursion.",
            "Write a Java Spliterator for a custom data structure to enable parallel stream processing.",
            # JPA and persistence
            "Write a JPA entity class with optimistic locking using @Version and a corresponding repository.",
            "Create a Spring Data JPA specification for dynamic query building with optional filters.",
            "Implement a JPA auditing configuration that automatically populates created/updated timestamps.",
            "Write a JPA JPQL query to fetch a parent entity with its children in a single query using JOIN FETCH.",
            "Create a JPA entity graph definition to control lazy/eager loading per use case.",
            "Implement a JPA converter for storing an enum as a specific string in the database.",
            "Write a Spring Data JPA repository method using @Query to compute aggregate statistics.",
            "Create a JPA entity with a polymorphic inheritance strategy using SINGLE_TABLE discriminator.",
            "Implement a JPA batch insert configuration using hibernate.jdbc.batch_size for bulk loading.",
            "Write a Flyway migration script for adding a nullable column and backfilling its values.",
            # Testing
            "Write a JUnit 5 parameterized test that verifies a sorting algorithm on multiple inputs.",
            "Create a Mockito-based unit test for a service that depends on a repository and an email sender.",
            "Implement a Spring Boot integration test using @SpringBootTest and TestRestTemplate.",
            "Write a JUnit 5 test extension that injects a mock clock for deterministic time-based tests.",
            "Create a WireMock stub server test for a service that calls an external REST API.",
            "Implement a Testcontainers-based integration test using a real PostgreSQL container.",
            "Write a JUnit 5 test that verifies concurrent access to a thread-safe data structure.",
            "Create an ArchUnit test that enforces layered architecture rules in a Spring Boot project.",
            "Implement a Kafka integration test using an embedded Kafka broker with consumer verification.",
            "Write a JUnit 5 test using @TempDir that verifies file I/O operations.",
            # Java core features
            "Implement a Java generic Pair class with equals, hashCode, and Comparable support.",
            "Write a Java sealed interface hierarchy modeling a result type with success and failure variants.",
            "Create a Java record class for an immutable value object with custom validation in the constructor.",
            "Implement a Java functional interface with default methods for composing transformations.",
            "Write a Java module-info.java for a multi-module project with explicit exports and requires.",
            "Create a Java ResourceBundle-based internationalization utility with fallback locale support.",
            "Implement a Java SPI (Service Provider Interface) plugin mechanism with dynamic loading.",
            "Write a Java reflection-based property copier that maps between DTOs and entities.",
            "Create a Java annotation processor that generates boilerplate code at compile time.",
            "Implement a Java serialization proxy pattern to control the serialized form of a class.",
            # Algorithms
            "Write a Java solution to implement a LRU cache using a LinkedHashMap.",
            "Create a Java implementation of a min-heap with insert, extractMin, and heapify.",
            "Implement a Java graph BFS and DFS that return traversal order from a given start node.",
            "Write a Java dynamic programming solution for the longest increasing subsequence.",
            "Create a Java implementation of the merge intervals problem sorting and merging overlapping ranges.",
            "Implement a Java trie for dictionary word search with prefix and exact match support.",
            "Write a Java solution to detect a cycle in a linked list using Floyd's algorithm.",
            "Create a Java implementation of quick select to find the kth largest element in O(n) average.",
            "Implement a Java sliding window algorithm to find the maximum sum subarray of size k.",
            "Write a Java solution for matrix spiral traversal returning elements in clockwise order.",
            # Build and tooling
            "Write a Maven POM file configuring Checkstyle, SpotBugs, and JaCoCo plugins.",
            "Create a Gradle build script for a multi-project Java build with shared dependency versions.",
            "Write a Dockerfile for a Spring Boot application with a JVM tuned for container environments.",
            "Create a GitHub Actions workflow for a Java project with build, test, and SonarCloud analysis.",
            "Implement a Maven plugin configuration for generating Java source from an OpenAPI spec.",
            # Additional Spring Boot
            "Write a Spring Boot GraphQL endpoint using Spring for GraphQL with a DataFetcher.",
            "Create a Spring Boot AMQP consumer that processes messages from a RabbitMQ queue with DLQ handling.",
            "Implement a Spring Boot health indicator that checks connectivity to an external service.",
            "Write a Spring Boot filter that injects a request-scoped correlation ID into the MDC.",
            "Create a Spring Boot configuration that conditionally enables a bean based on a profile.",
            "Implement a Spring Boot REST endpoint with ETag support for conditional GET requests.",
            "Write a Spring Boot service using WebClient for non-blocking HTTP calls to an external API.",
            "Create a Spring Boot security configuration using method-level @PreAuthorize annotations.",
            "Implement a Spring Boot task that runs on application startup to seed reference data.",
            "Write a Spring Boot endpoint that accepts a file upload and streams it to object storage.",
            "Create a Spring Boot metrics configuration using Micrometer with Prometheus export.",
            "Implement a Spring Boot session management configuration using Redis for distributed sessions.",
            "Write a Spring Boot async method using @Async and a configured ThreadPoolTaskExecutor.",
            "Create a Spring Boot HATEOAS response that includes hypermedia links for related resources.",
            "Implement a Spring Boot REST controller with content negotiation for JSON and XML.",
            # Additional concurrency
            "Write a Java CompletableFuture chain that fans out to 5 services and combines all results.",
            "Implement a Java rate limiter using a token bucket with atomic operations.",
            "Create a Java function using ForkJoinPool to compute a recursive parallel sum.",
            "Write a Java BlockingDeque-based priority task queue with a bounded capacity.",
            "Implement a Java ReadWriteLock-protected in-memory registry with snapshot capability.",
            "Create a Java ScheduledExecutorService wrapper with structured error handling per task.",
            "Write a Java virtual thread server that handles 10,000 concurrent connections.",
            "Implement a Java phaser-based multi-phase parallel computation workflow.",
            "Create a Java exchanger-based pipeline where two threads swap data buffers.",
            "Write a Java lock-free linked queue implementing Michael-Scott algorithm.",
            # Additional collections and streams
            "Write a Java Stream pipeline that computes a histogram from a list of measurements.",
            "Create a Java Collector that partitions a stream into chunks of fixed size.",
            "Implement a Java function using Stream.iterate to lazily generate a recurrence sequence.",
            "Write a Java function using Collectors.groupingBy with downstream counting and sorting.",
            "Create a Java function using IntStream.range to compute matrix row sums without loops.",
            "Implement a Java function using Stream.flatMap to flatten a nested list structure.",
            "Write a Java function using Collectors.toUnmodifiableMap with a merge function.",
            "Create a Java function using Stream.takeWhile and dropWhile for conditional prefix processing.",
            "Implement a Java Comparator.comparing chain for sorting by three fields with null safety.",
            "Write a Java function using Map.Entry streams to invert a frequency map.",
            # Additional JPA and persistence
            "Write a JPA entity with a composite primary key using @EmbeddedId.",
            "Create a Spring Data JPA projection interface for a read-only DTO query.",
            "Implement a JPA second-level cache configuration using Ehcache.",
            "Write a JPA criteria query builder for a dynamic multi-field search form.",
            "Create a Spring Data JPA repository with a custom @Modifying bulk update query.",
            "Implement a JPA entity listener that computes a hash of audited fields on pre-update.",
            "Write a Spring Data JPA method using @Lock for a pessimistic read lock.",
            "Create a JPA entity with a one-to-many self-referential relationship for categories.",
            "Implement a JPA multi-tenancy configuration using schema-based tenant isolation.",
            "Write a Spring Data repository that uses QueryDSL predicates for composable filtering.",
            # Additional testing
            "Write a JUnit 5 extension that resets a singleton between tests.",
            "Create a Mockito ArgumentCaptor test that verifies the correct object is passed to a mock.",
            "Implement a Spring Boot slice test using @DataJpaTest with an H2 in-memory database.",
            "Write a JUnit 5 test using @MethodSource to supply test cases from a factory method.",
            "Create a REST-assured integration test for a Spring Boot API with authentication.",
            "Implement a JUnit 5 test that verifies an exception is thrown with a specific message.",
            "Write a Testcontainers test that uses a PostgreSQL container with Flyway migrations.",
            "Create a JUnit 5 test that measures performance and fails if it exceeds a time limit.",
            "Implement a Spring Boot MockMvc test for a multipart file upload endpoint.",
            "Write a Mutation testing configuration using PIT (pitest) for a Java service class.",
            # Additional algorithms
            "Write a Java implementation of a red-black tree with insert and delete operations.",
            "Create a Java solution for the word break problem using dynamic programming.",
            "Implement a Java function that computes the number of islands using DFS on a grid.",
            "Write a Java solution for validating a Sudoku board without solving it.",
            "Create a Java implementation of the Boyer-Moore-Horspool string search algorithm.",
            "Implement a Java function that finds the shortest path in a maze using BFS.",
            "Write a Java solution for the coin change problem using bottom-up dynamic programming.",
            "Create a Java implementation of a consistent hash ring for distributed cache routing.",
            "Implement a Java function that detects and returns all strongly connected components using Tarjan's algorithm.",
            "Write a Java function that computes the edit distance between two strings iteratively.",
            # Additional patterns and best practices
            "Write a Java sealed class hierarchy modeling a Result<T, E> type with map and flatMap.",
            "Create a Java functional interface composition chain for a validation pipeline.",
            "Implement a Java record with custom equals and hashCode overrides for value semantics.",
            "Write a Java annotation-driven validation framework using reflection to apply constraints.",
            "Create a Java module system configuration with a multi-module Maven project.",
            "Implement a Java SPI loader that discovers and ranks service implementations by priority.",
            "Write a Java WeakHashMap-based memoization cache that allows GC to reclaim entries.",
            "Create a Java fluent API for building SQL WHERE clauses with type-safe predicates.",
            "Implement a Java dynamic proxy that logs all method calls on an interface.",
            "Write a Java ByteBuddy agent that instruments a method to record its execution time.",
            "Create a Java function that computes a SHA-256 HMAC for API request signing.",
            "Implement a Java function that parses and validates a JSON Web Token without external libraries.",
            "Write a Java function that generates a ULID (universally unique lexicographically sortable ID).",
            "Create a Java function that implements a consistent hashing ring for distributed cache lookup.",
            "Implement a Java function that computes the Soundex phonetic code for a name string.",
            "Write a Java function that performs Bloom filter membership testing with configurable false positive rate.",
            "Create a Java function that implements LFU (Least Frequently Used) cache eviction policy.",
            "Implement a Java function that converts an adjacency list graph to a DOT format string.",
            "Write a Java function that parses a robots.txt file and answers whether a URL is allowed.",
            "Create a Java function that converts Roman numerals to integers and back.",
            "Implement a Java function that generates a permutation of a list using Heap's algorithm.",
            "Write a Java function that computes the Jaccard similarity between two sets of strings.",
            "Create a Java function that implements a fixed-size sliding window for stream statistics.",
            "Implement a Java function that converts a POJO to a Map using reflection.",
            "Write a Java function that finds all permutations of a string with no repeated characters.",
            "Create a Java function that validates and formats a semantic version string.",
            "Implement a Java function that computes the strongly connected components of a directed graph.",
            "Write a Java function that computes the maximum subarray sum using Kadane's algorithm.",
            "Create a Java function that solves the word ladder problem using bidirectional BFS.",
            "Implement a Java function that computes the minimum spanning tree using Prim's algorithm.",
            "Write a Java function that decodes a URL-encoded query string into a key-value map.",
            "Create a Java function that implements a skip list for O(log n) average-case insertion.",
            "Implement a Java function that solves a maze using recursive DFS and returns the path.",
            "Write a Java function that generates all subsets of a set using bitmask enumeration.",
            "Create a Java function that computes the number of ways to make change for a target amount.",
        ],
    },
    "html_css": {
        "topic": "html_css",
        "followup_pool": ADVERSARIAL_FOLLOWUP_POOL["html_css"],
        "root_tasks": [
            # Layout: Flexbox
            "Create a CSS Flexbox navigation bar with a logo on the left, links in the center, and a CTA button on the right.",
            "Build a responsive card grid using Flexbox that shows 4 columns on desktop, 2 on tablet, and 1 on mobile.",
            "Write CSS for a Flexbox sticky footer layout where the content grows to fill the available space.",
            "Create a Flexbox holy grail layout with header, footer, left sidebar, main content, and right sidebar.",
            "Build a Flexbox pricing table with three plan cards that align features and CTAs at the bottom.",
            "Write CSS for a Flexbox media object pattern with an image on the left and text on the right.",
            "Create a Flexbox tab bar that distributes tabs evenly and highlights the active tab.",
            "Build a Flexbox form layout where labels and inputs align in a two-column grid.",
            "Write CSS for a Flexbox avatar group that overlaps profile pictures with a count badge.",
            "Create a Flexbox loading skeleton component that matches the dimensions of a card layout.",
            # Layout: Grid
            "Create a CSS Grid dashboard layout with a sidebar, header, main area, and footer using named template areas.",
            "Build a CSS Grid magazine-style article layout with a large hero image and sidebar text.",
            "Write CSS for a CSS Grid photo gallery with masonry-like variable-height items.",
            "Create a CSS Grid calendar view showing 7 columns with proper day alignment for any month.",
            "Build a CSS Grid product listing page with 3 columns desktop, 2 tablet, 1 mobile, and a full-width featured item.",
            "Write CSS for a CSS Grid timeline component with alternating left/right event cards.",
            "Create a CSS Grid email client layout with a resizable folder list, email list, and preview pane.",
            "Build a CSS Grid periodic table layout with correct chemical family color coding.",
            "Write CSS for a CSS Grid resume layout that prints cleanly on A4 paper.",
            "Create a CSS Grid kanban board with fixed column widths and scrollable card lists.",
            # Responsive design
            "Write a responsive HTML page with a hamburger menu that toggles a mobile navigation drawer.",
            "Create a responsive hero section with background image, overlay text, and a CTA that adapts to mobile.",
            "Build a responsive data table that collapses into stacked card cards on small screens.",
            "Write CSS for a responsive image comparison slider with a draggable divider.",
            "Create a responsive multi-column text layout using CSS columns with proper hyphenation.",
            "Build a responsive video embed that maintains 16:9 aspect ratio on all screen sizes.",
            "Write CSS for a responsive map container that fills available width with a fixed aspect ratio.",
            "Create a responsive testimonials carousel with touch-swipe support using pure CSS.",
            "Build a responsive admin sidebar that collapses to an icon-only mode on smaller screens.",
            "Write CSS media queries for a print stylesheet that hides navigation and formats content for printing.",
            # Animations and transitions
            "Create a CSS keyframe animation for a loading spinner with smooth rotation.",
            "Write CSS for a button that animates a progress bar fill on hover.",
            "Create a CSS transition for a card flip animation revealing the back face on hover.",
            "Build a CSS animated typewriter effect that types and erases text on a loop.",
            "Write CSS keyframe animations for a skeleton loader that shimmers with a gradient sweep.",
            "Create a CSS scroll-triggered fade-in animation using the Intersection Observer API.",
            "Build a CSS animated hamburger menu icon that morphs into an X on activation.",
            "Write CSS for a confetti animation using multiple keyframes and random delays.",
            "Create a CSS animated notification badge that pulses to indicate new items.",
            "Build a CSS page transition animation that fades between views using CSS variables.",
            # Forms and UI components
            "Build an HTML form with custom-styled checkboxes, radio buttons, and a toggle switch using only CSS.",
            "Create a CSS-only accordion that expands and collapses sections without JavaScript.",
            "Write CSS for a custom file input that displays the selected filename.",
            "Create a styled HTML range slider with a tooltip showing the current value.",
            "Build an HTML select element replaced with a custom dropdown using CSS and minimal JS.",
            "Write CSS for a multi-step form with an animated progress indicator between steps.",
            "Create a floating label form field where the label animates above the input on focus.",
            "Build a password strength meter with color-coded segments and a requirement checklist.",
            "Write CSS for an autocomplete input with a styled suggestions dropdown.",
            "Create a multi-select tag input where typed text becomes a removable tag on Enter.",
            # Typography
            "Write CSS for a fluid typography scale using clamp() that interpolates between mobile and desktop sizes.",
            "Create a CSS dropcap style for article first paragraphs with proper fallback.",
            "Build a CSS text truncation component that shows an expand/collapse button after 3 lines.",
            "Write CSS for a pull-quote component with a decorative quotation mark and citation.",
            "Create CSS styles for a code block with syntax highlighting classes and a copy button.",
            "Build a CSS reading progress indicator that fills a thin bar at the top as the user scrolls.",
            "Write CSS for a highlighted search result component that wraps matched text in a mark element.",
            "Create a CSS font-face declaration with woff2 and woff fallbacks and proper font-display swap.",
            "Build CSS styles for a badge component with a variety of color variants using CSS custom properties.",
            "Write CSS for a numbered list with custom counters and decorative connector lines.",
            # Theming and CSS variables
            "Create a CSS custom property system for a design token library with color, spacing, and typography scales.",
            "Build a theme switcher that toggles between light and dark mode using CSS variables and JS.",
            "Write CSS for a component that inherits a configurable accent color from a parent CSS variable.",
            "Create a CSS grid of color swatches generated from a set of CSS custom properties.",
            "Build a CSS variable-based spacing scale and apply it consistently to a card component.",
            "Write CSS for a glassmorphism card using backdrop-filter and CSS variables for easy theming.",
            "Create a CSS custom property cascade that allows per-component overrides of global tokens.",
            "Build a dynamic theme generator that applies user-chosen colors as CSS variables.",
            "Write CSS variables for a shadow elevation system with five levels like Material Design.",
            "Create a CSS utility class system (like Tailwind) using CSS variables for spacing and color.",
            # Accessibility
            "Write HTML and CSS for a skip-to-content link that appears on keyboard focus.",
            "Create an accessible modal dialog with ARIA roles, focus management, and keyboard dismissal.",
            "Build an accessible dropdown navigation menu with ARIA expanded/collapsed states.",
            "Write CSS for a focus indicator style that passes WCAG 2.1 AA contrast requirements.",
            "Create an accessible tab interface with ARIA tablist, tab, and tabpanel roles and keyboard navigation.",
            "Build an accessible autocomplete widget following the ARIA authoring practices guide.",
            "Write HTML for an accessible data table with proper scope attributes and a caption.",
            "Create a visually hidden CSS class and use it to add descriptive text for screen readers.",
            "Build an accessible star rating widget using radio inputs styled as visual stars.",
            "Write CSS for a high-contrast mode using prefers-contrast media query.",
            # Performance and advanced CSS
            "Write CSS that uses content-visibility to improve rendering performance on a long list.",
            "Create a CSS containment strategy using contain: layout style that prevents style recalculation bleed.",
            "Build a critical CSS extraction configuration for a landing page to eliminate render-blocking styles.",
            "Write CSS custom properties with @property declarations for animatable gradient transitions.",
            "Create a CSS grid layout that avoids layout thrashing by minimizing forced synchronous layouts.",
            "Write CSS using will-change property hints for animations that benefit from GPU compositing.",
            "Create a CSS layer system using @layer to manage specificity in a design system.",
            "Build a CSS subgrid layout where child elements align to the parent grid tracks.",
            "Write CSS using container queries to change a card layout based on its container width.",
            "Create a CSS scroll-snap layout for a full-page scrolling presentation.",
            # Miscellaneous
            "Write HTML meta tags for proper Open Graph and Twitter Card social sharing previews.",
            "Create a favicon.ico and apple-touch-icon HTML setup with proper sizes and link tags.",
            "Build an HTML structured data markup using Schema.org JSON-LD for a product page.",
            "Write CSS for a print layout that formats a resume document with proper page breaks.",
            "Create an HTML email template compatible with Gmail, Outlook, and Apple Mail using table layout.",
            "Build a CSS grid of social sharing buttons with consistent icon sizing and hover states.",
            "Write HTML and CSS for a cookie consent banner that slides up from the bottom.",
            "Create a CSS animated scroll indicator arrow that bounces below a hero section.",
            "Build an HTML progress element with a custom CSS style that animates on load.",
            "Write CSS for a comparison table with sticky first column and highlighted recommended plan.",
            # Additional layout: advanced flexbox
            "Create a Flexbox masonry-inspired layout where items fill columns top-to-bottom.",
            "Build a Flexbox split-screen hero section with two equal halves that stack on mobile.",
            "Write CSS for a Flexbox chip/tag component that wraps gracefully with a max-width.",
            "Create a Flexbox sidebar that is collapsible with a transition and pushes content to the right.",
            "Build a Flexbox stepper component with connected lines between step indicators.",
            "Write CSS for a Flexbox toolbar with left group, center title, and right action buttons.",
            "Create a Flexbox nested comment indentation layout with connecting lines.",
            "Build a Flexbox equal-height card layout that stretches buttons to the bottom of each card.",
            "Write CSS for a Flexbox overlay image caption that slides up from the bottom on hover.",
            "Create a Flexbox scrollable horizontal tag list with fade gradients at both ends.",
            # Additional layout: advanced grid
            "Build a CSS Grid layout with named lines for precise alignment of overlapping elements.",
            "Write CSS for a CSS Grid dense auto-placement layout for a photo mosaic.",
            "Create a CSS Grid layout where some items span multiple columns and rows for emphasis.",
            "Build a CSS Grid ASCII art layout that positions elements into a precise visual pattern.",
            "Write CSS for a CSS Grid responsive dashboard that rearranges widgets at three breakpoints.",
            "Create a CSS Grid list layout that switches between vertical list and horizontal row.",
            "Build a CSS Grid newspaper front page with a large lead story and smaller sidebars.",
            "Write CSS for a CSS Grid footer with four equal columns collapsing to two then one.",
            "Create a CSS Grid split sidebar layout where sidebar width adjusts with a CSS variable slider.",
            "Build a CSS Grid product detail page with image gallery, title, and add-to-cart side by side.",
            # Additional responsive design
            "Write CSS container queries that change a component layout based on its wrapper width.",
            "Create a responsive HTML email layout compatible with Outlook using table fallbacks.",
            "Build a responsive image srcset configuration for high-DPI displays and mobile.",
            "Write CSS for an adaptive sidebar that collapses to a bottom navigation bar on mobile.",
            "Create a responsive data visualization wrapper that hides columns on small screens.",
            "Build a CSS clamp-based fluid layout that needs no media query breakpoints.",
            "Write CSS for a responsive hero that shows a different background image per breakpoint.",
            "Create a responsive modal that becomes a full-screen drawer on mobile.",
            "Build a CSS sticky table of contents that tracks the active section on scroll.",
            "Write CSS for a responsive comparison slider with touch drag support using CSS custom properties.",
            # Additional animations
            "Create a CSS animation for a wave-like text loading effect on a heading.",
            "Write CSS keyframes for a morphing SVG icon transition between two states.",
            "Build a CSS animated progress circle using stroke-dasharray and SVG.",
            "Create a CSS particle burst animation triggered by a button click using keyframes.",
            "Write CSS for a typing cursor animation that blinks at a configurable rate.",
            "Build a CSS counter animation that counts from 0 to a target value on page load.",
            "Create a CSS ripple effect that expands from the click point of a button.",
            "Write CSS for a card entrance animation using staggered delays for a grid of cards.",
            "Build a CSS animated border gradient that rotates around a card component.",
            "Create a CSS scroll-driven animation that changes a header background on scroll using @scroll-timeline.",
            # Additional forms and interaction
            "Write CSS for a search bar that expands with a smooth transition when focused.",
            "Build a CSS-only star rating input using radio buttons and the general sibling combinator.",
            "Create a CSS quantity input with plus and minus buttons styled consistently.",
            "Write CSS for a segmented control component (like iOS UISegmentedControl) using radio buttons.",
            "Build a CSS form field with a character count indicator that turns red near the limit.",
            "Create a CSS combobox component with a dropdown attached below the input.",
            "Write CSS for a rich text toolbar with icon buttons and active state highlighting.",
            "Build a CSS slider component with a colored filled track reflecting the current value.",
            "Create a CSS OTP input with auto-advance styling for a one-time password entry.",
            "Write CSS for a drag-and-drop file zone with dashed border and hover highlight.",
            # Additional SVG and icons
            "Write an inline SVG icon system that uses a sprite sheet and references icons by ID.",
            "Create an SVG animated loading bar that fills and empties in a loop.",
            "Build an SVG pie chart with configurable slice angles and hover tooltips.",
            "Write CSS to animate an SVG path drawing effect on page load using stroke-dashoffset.",
            "Create an SVG wave divider between two sections with configurable amplitude and frequency.",
            "Build an inline SVG spinner with rotating gradient using conic-gradient.",
            "Write CSS for an SVG icon that changes color and size on hover with transition.",
            "Create a CSS mask using an SVG shape to clip an image into a custom silhouette.",
            "Build an SVG progress ring that animates to a percentage value on component mount.",
            "Write CSS for an SVG background pattern that tiles across a section.",
            # Additional theming and tokens
            "Build a CSS design token system using @layer and CSS custom properties for a component library.",
            "Write CSS using @property to define an animatable custom property with type and initial value.",
            "Create a CSS nesting-based component stylesheet using the native CSS nesting syntax.",
            "Build a CSS cascade layer architecture with base, components, and utilities layers.",
            "Write CSS logical properties for a layout that supports both LTR and RTL text direction.",
            "Create a CSS variable-driven component that is themeable by just overriding a few root tokens.",
            "Build a CSS color scheme that automatically switches between light and dark based on OS preference.",
            "Write CSS for a scoped theme that overrides global tokens for just a specific section.",
            "Create a CSS font-size scale using CSS custom properties and rem units.",
            "Build a CSS z-index management system using named custom properties.",
            # Additional advanced CSS features
            "Write CSS using @supports to provide a modern layout with a fallback for older browsers.",
            "Create a CSS scroll-snap container for a vertical one-page scrolling website.",
            "Build a CSS subgrid layout where child grid items align to grandparent grid tracks.",
            "Write CSS for a sticky positioned element within a scroll-snap container.",
            "Create a CSS paint worklet (Houdini) that generates a custom background pattern.",
            "Build a CSS layout using aspect-ratio to maintain proportional elements without JS.",
            "Write CSS using overscroll-behavior to prevent scroll chaining in a modal.",
            "Create CSS for a multi-column text layout with a drop cap and pull quote float.",
            "Build a CSS anchor positioning layout (CSS Anchor Positioning API) for a tooltip.",
            "Write CSS for a view transition animation between two page states using the View Transitions API.",
            "Build a CSS tooltip that positions itself above, below, left, or right using CSS anchor positioning.",
            "Create a CSS card component with an image header, body text, and a footer that pins to the bottom.",
            "Write CSS for a responsive pricing toggle that switches between monthly and annual pricing.",
            "Build a CSS timeline with vertical connector lines and alternating event cards.",
            "Create CSS for a notification dropdown that pops up from a bell icon with a badge count.",
            "Write CSS for a status indicator dot with a pulsing animation for live status.",
            "Build a CSS skill bar component that animates to a percentage width on page load.",
            "Create CSS for a breadcrumb trail with chevron separators and a truncated middle section.",
            "Write CSS for a tag cloud where font size reflects tag frequency.",
            "Build a CSS media player controls bar with a progress scrubber and volume slider.",
            "Create CSS for a chat bubble component with a tail arrow and alternating alignment.",
            "Write CSS for a responsive login card centered in the viewport with a logo above.",
            "Build a CSS data badge that overlays a corner of an element with a number count.",
            "Create CSS for a full-screen search overlay that fades in over the page.",
            "Write CSS for a collapsible FAQ section using the details/summary HTML elements.",
            "Build a CSS photo caption overlay that appears on image hover with a smooth fade.",
            "Create CSS for a floating help button that stays fixed in the bottom-right corner.",
            "Write CSS for an error boundary component with a warning icon and retry button.",
            "Build a CSS skeleton loader that matches the exact shape of a profile card.",
            "Create CSS for a side drawer that slides in from the left with a dark overlay backdrop.",
        ],
    },
}

# ── General knowledge follow-up pool ────────────────────────────────────────────────────────

GENERAL_FOLLOWUP_POOL = [
    "Can you expand on that with more specific details and concrete examples?",
    "What are the most common misconceptions people have about this topic?",
    "How does this compare to related concepts or alternatives?",
    "Summarize the key takeaways as a structured bullet-point guide.",
    "How would an expert in this field explain it differently to a beginner?",
    "What are the practical real-world applications or implications of this?",
    "Can you provide a step-by-step breakdown or a concrete case study?",
    "What recent developments or research have changed our understanding of this?",
]


def load_quora_questions(base_dir):
    """Load Quora questions from existing standard and large prompts files."""
    questions = []
    search_paths = [
        os.path.join(base_dir, "prompts", "standard", "prompts.json"),
    ]
    large1_dir = os.path.join(base_dir, "prompts", "large1")
    if os.path.isdir(large1_dir):
        for fname in os.listdir(large1_dir):
            if fname.endswith(".json"):
                search_paths.append(os.path.join(large1_dir, fname))

    seen = set()
    for path in search_paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for q in data.get("negative", {}).get("prompts", []):
            if q not in seen:
                seen.add(q)
                questions.append(q)

    return questions


def build_session(session_id, topic, root_task, followup_pool, rng):
    """Build a single session dict with 3-5 randomly ordered follow-up turns."""
    num_turns = rng.randint(3, 5)
    num_followups = num_turns - 1
    chosen = rng.sample(followup_pool, min(num_followups, len(followup_pool)))
    rng.shuffle(chosen)
    return {
        "session_id": session_id,
        "topic": topic,
        "turns": [root_task] + chosen,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate conversation prompt templates.")
    parser.add_argument(
        "--output",
        default=os.path.join("prompts", "conversations", "conversations.json"),
        help="Output JSON file path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--neg-general-count",
        type=int,
        default=9000,
        help="Number of negative general knowledge sessions.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # ── Positive sessions ────────────────────────────────────────────────────
    print(f"Building {len(POSITIVE_SESSIONS_DATA)} positive Python DS sessions...")
    positive_sessions = []
    for i, entry in enumerate(POSITIVE_SESSIONS_DATA):
        session_id = f"pos_{i + 1:03d}"
        session = build_session(
            session_id=session_id,
            topic="python_data_science",
            root_task=entry["root"],
            followup_pool=entry["followups"],
            rng=rng,
        )
        positive_sessions.append(session)

    # ── Negative general sessions ────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    quora_questions = load_quora_questions(base_dir)
    print(f"Loaded {len(quora_questions)} Quora questions.")

    if len(quora_questions) < args.neg_general_count:
        print(
            f"Warning: only {len(quora_questions)} questions available, "
            f"requested {args.neg_general_count}. Using all available."
        )
        args.neg_general_count = len(quora_questions)

    sampled_questions = rng.sample(quora_questions, args.neg_general_count)
    print(f"Building {args.neg_general_count} negative general sessions...")
    neg_general_sessions = []
    for i, question in enumerate(sampled_questions):
        session_id = f"neg_gen_{i + 1:05d}"
        session = build_session(
            session_id=session_id,
            topic="general_knowledge",
            root_task=question,
            followup_pool=GENERAL_FOLLOWUP_POOL,
            rng=rng,
        )
        neg_general_sessions.append(session)

    # ── Negative adversarial code sessions ───────────────────────────────────
    neg_code_sessions = []
    for theme_key, theme_data in ADVERSARIAL_THEMES.items():
        root_tasks = theme_data["root_tasks"]
        followup_pool = theme_data["followup_pool"]
        topic = theme_data["topic"]
        print(f"Building {len(root_tasks)} negative code sessions for theme '{theme_key}'...")
        for i, root_task in enumerate(root_tasks):
            session_id = f"neg_{theme_key}_{i + 1:03d}"
            session = build_session(
                session_id=session_id,
                topic=topic,
                root_task=root_task,
                followup_pool=followup_pool,
                rng=rng,
            )
            neg_code_sessions.append(session)

    # ── Assemble output ──────────────────────────────────────────────────────
    output = {
        "positive": {
            "repeat": 100,
            "sessions": positive_sessions,
        },
        "negative_general": {
            "repeat": 1,
            "sessions": neg_general_sessions,
        },
        "negative_code": {
            "repeat": 1,
            "sessions": neg_code_sessions,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_sessions = len(positive_sessions) + len(neg_general_sessions) + len(neg_code_sessions)
    estimated_captures = (
        len(positive_sessions) * output["positive"]["repeat"]
        + len(neg_general_sessions) * output["negative_general"]["repeat"]
        + len(neg_code_sessions) * output["negative_code"]["repeat"]
    )
    print(f"\nDone. Written to: {args.output}")
    print(f"  Positive sessions:        {len(positive_sessions):>6}")
    print(f"  Negative general sessions:{len(neg_general_sessions):>6}")
    print(f"  Negative code sessions:   {len(neg_code_sessions):>6}")
    print(f"  Total session templates:  {total_sessions:>6}")
    print(f"  Estimated total captures: {estimated_captures:>6}")


if __name__ == "__main__":
    main()
