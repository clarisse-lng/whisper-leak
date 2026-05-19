# LGBM Changes

## utils.py

Added import:

In `ModelTrainer.fit()`:
```python
if isinstance(self.model, LightGBMClassifier):
# Handle LightGBM models (using DataFrames)
  self.model.fit(
    train_df=train_data.df,
    val_df=val_data.df,
    patience=self.config.patience
)
# Create basic history for consistency
    self.history = {
      'best_epoch': getattr(self.model.model, 'best_iteration_', 0),
      'train_losses': [],
      'val_losses': [],
      'train_accs': [],
      'val_accs': [],
    }
```

In `ModelTrainer.predict()`:
```python
if isinstance(self.model, LightGBMClassifier):
  # For LightGBM, data should be a DataFrame
  data = data.df
            
  # Get predictions
  try:
    scores = self.model.predict_proba(data)
    labels = data['target'].values
    return scores, labels, None  # Loss not available for LightGBM
  except Exception as e:
    PrintUtils.print_extra(f"Error during LightGBM prediction: {e}")
    return None, None, None
```


In `get_prediction_scores()` :
```python
    if isinstance(model, LightGBMClassifier):
        PrintUtils.print_extra("Getting predictions using LightGBMClassifier predict_proba")
        if dataloader_or_df is None:
             PrintUtils.print_extra("Error: dataloader_or_df is required for LightGBMClassifier prediction.")
             return None, None, None
        if 'target' not in dataloader_or_df.columns:
            PrintUtils.print_extra("Error: dataloader_or_df must contain 'target' column for evaluation.")
            return None, None, None

        try:
            # Note: LightGBM predict_proba uses the internal _prepare_features
            # which expects the original 'time_diffs', 'data_lengths' columns.
            # Ensure dataloader_or_df has these columns in the correct format.
             all_scores = model.predict_proba(dataloader_or_df)
             all_labels = dataloader_or_df['target'].values
             loss = np.nan # LightGBM doesn't compute loss in the same way during predict
             PrintUtils.print_extra(f"Generated {len(all_scores)} predictions from LightGBM.")
             return np.array(all_scores), np.array(all_labels), loss
        except Exception as e:
             PrintUtils.print_extra(f"Error during LightGBM prediction: {e}")
             import traceback
             traceback.print_exc()
             return None, None, None
```



---

## whisper_leak_train.py
In `create_model()` — added case alongside LGBM:
```python
elif model_type == 'LGBM':
        model = LightGBMClassifier(norm)
        model_path = os.path.join(models_dir, 'lightgbm_binary_classifier.pth')
```

---

## Run command

```bash
python3 whisper_leak_train.py -c GPT4o -m LGBM -i data/main/gpt4o -p prompts/standard/prompts.json
```
