# Taxi ETA Prediction — Solution Writeup

---

## Final Score

Dev MAE: 263.3 seconds

---

## Approach (Overview)

I began with exploratory data analysis to understand feature distributions, temporal patterns, and anomalies in the dataset. Based on insights from prior research papers on NYC Taxi ETA prediction (https://www.ijraset.com/research-paper/new-york-city-taxi-trip-duration-prediction-using-machine-learning) (https://sdaulton.github.io/TaxiPrediction/) (https://norma.ncirl.ie/6268/1/janvirajeshrajani.pdf), I experimented with both deep learning approaches (Deep & Cross Networks) and tree-based models.

I engineered approximately 32 features capturing spatial (zone distances), temporal (hour, weekday effects), Haversine distance, angles between the features and demand-related signals. Feature importance was evaluated using correlation analysis and SHAP values. To improve generalization and efficiency, I applied a combination of filter and wrapper-based feature selection techniques to reduce redundancy and prevent data leakage.

I trained multiple models including Decision Trees, CatBoost, Deep & Cross Networks (DCN), and XGBoost. Hyperparameter tuning was performed using Optuna.

XGBoost delivered the best performance, achieving a Dev MAE of 263.3 seconds while staying within the 2-2.5GB Docker memory constraint.

Special attention was given to handling anomalies such as COVID-period distribution shifts and noisy outliers.

---

## What Didn’t Work

1. Deep & Cross Networks (DCN)  
   While effective at modeling feature interactions, DCNs underperformed compared to tree-based methods on this tabular dataset. They also required higher computational resources without proportional gains.

2. Complex Ensembles  
   Initial attempts at combining multiple models increased training time significantly but did not yield meaningful improvements over a well-tuned XGBoost model.

3. Using All Features  
   Retaining all engineered features led to slight overfitting and increased inference cost. Feature selection improved both performance and efficiency.

---

## Where AI Tooling Helped Most

Tools used: Gemini, ChatGPT and Cursor

AI tools significantly accelerated:
- Thinking out loud and validating ideas
- Feature engineering ideation (spatial, temporal, and interaction features)
- Setting up hyperparameter tuning pipelines using Optuna
- Debugging preprocessing and training workflows
- Translating research ideas into working implementations

Limitations:
- Unable to identify dataset-specific anomalies without manual inspection
- Limited effectiveness in fine-tuning beyond baseline performance

---

## Next Experiments

- Incorporate graph-based spatial features using road network information
- Evaluate LightGBM with advanced categorical feature handling
- Implement time-aware validation using rolling windows
- Train multiple models on different temporal segments and ensemble them
- Integrate external data sources such as weather or traffic conditions

---

## Reproducibility

### Local Environment
1. **Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. **Data**: Download the NYC TLC data and local context:
   ```bash
   python data/download_data.py
   python data/download_extras.py
   ```
3. **Train**: Run the optimized XGBoost training pipeline (includes feature engineering and Optuna tuning):
   ```bash
   python train.py
   ```
4. **Grade**: Evaluate on the local dev set:
   ```bash
   python grade.py
   ```

### Docker Environment (Recommended)
The included `Dockerfile` is optimized to run on any machine under the 2.5GB size constraint.
1. **Build**:
   ```bash
   docker build -t eta-challenge .
   ```
2. **Train**:
   ```bash
   docker run --rm -v $(pwd)/data:/app/data eta-challenge train.py
   ```
3. **Grade**:
   ```bash
   docker run --rm -v $(pwd)/data:/app/data eta-challenge grade.py
   ```
4. **Test**:
   ```bash
   docker run --rm eta-challenge -m pytest tests/
   ```

---

## Total Time Spent

Approximately 28 hours

---

## Key Takeaway

For structured, tabular problems under resource constraints, well-engineered features combined with optimized gradient boosting models consistently outperform more complex deep learning approaches.

---

## Additional Notes

- All experiments were conducted under strict memory constraints (2-2.5GB Docker environment), influencing model selection and optimization strategy.
- Training was performed iteratively with careful monitoring of performance vs compute trade-offs.
- Feature engineering and selection contributed more to performance gains than increasing model complexity.
- XGBoost provided the best balance of accuracy, training time, and deployment feasibility.

---