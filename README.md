
1. Introduction
Objective:  
This project aims to develop a machine-learning model capable of predicting house prices based on various properties-related features, such as size, location, and the number of rooms. Accurate house price predictions can aid buyers, sellers, and real estate professionals make informed decisions.
Dataset:
I used the "House Prices - Advanced Regression Techniques" dataset from Kaggle, which contains information about properties in Melbourne, Australia. Key features include the number of rooms, property type, location, size, and sale price.

2. Methodology	

The project was executed in five main phases:
Phase 1: Data Collection and Preparation
Data Loading: The dataset was loaded into a Pandas DataFrame for processing.
Data inspection and Cleaning: Initial inspection revealed missing values in several features. Missing values were handled appropriately using mean imputation for numerical features and mode imputation for categorical ones.
Data Cleaning: We removed irrelevant features, such as `Address`, and handled outliers by capping them at a reasonable percentile range.
Phase 2: Exploratory Data Analysis (EDA)
Distribution Analysis: Histograms and box plots were used to examine the distribution of key features like `Price`, `Building Area`, and `Distance`.
Feature Relationships: Scatter plots and correlation matrices highlighted relationships between features and the target variable (`Price`).
Outlier Detection: Box plots and scatter plots helped identify and handle outliers that could skew the model's predictions.
Phase 3: Feature Engineering
New Features: Created new features, such as the `Price per Square Meter`, to capture additional information that could improve the model's predictive power.
Encoding Categorical Variables: One-hot encoding was applied to categorical variables like `Type` and `Region_name`.
Normalisation: Numerical features were standardised to ensure the model treated all features equally.
Phase 4: Model Training and Evaluation
Model Selection: Several models were evaluated, including Linear Regression (with Ridge regularisation), Decision Trees, and Random Forests.
Model Evaluation Metrics: Performance was measured using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² score.
Hyperparameter Tuning: Randomized Search was used for hyperparameter tuning of Ridge, Lasso, Decision Tree, and Random Forest models.
Final Model Selection: The Random Forest model with tuned parameters performed the best, showing the balance between accuracy and robustness against overfitting.

Phase 5: Model Interpretation and Reporting
Feature Importance: Analyzed the importance scores of features in the best-performing Random Forest model to identify the most influential factors affecting house prices.
Visualisations: Created plots such as feature importance bar plots, scatter plots of predicted vs. actual prices, and partial dependence plots to illustrate the model's behaviour and performance.

Project Environment and Libraries
For this project, a robust environment was set up using Python 3.8.19. The following libraries and their specific versions were utilized to carry out data analysis, visualization, and model development:
Jupyter Notebook: Version 7.2.2, used for organizing the project into a structured workflow and for combining code execution with rich text.
Matplotlib: Version 3.3.3, employed for creating static, interactive, and animated visualizations to explore data trends.
NumPy: Version 1.19.5, utilized for numerical operations and handling array data structures efficiently.
Pandas: Version 1.2.0, used for data manipulation and analysis, including loading the dataset, cleaning, and preparing data for modelling.
Scikit-learn: Version 0.24.0, the primary machine learning library used for building and evaluating predictive models.
Seaborn: Version 0.11.0, used for statistical data visualization, enabling insightful plots and relationships between variables.
Typing-Extensions: Version 3.7.4.3, utilized for enhancing type hinting and annotations in Python, aiding in code clarity and maintenance.
This setup ensured a consistent and reproducible environment for data analysis and model development.

3. Results
 Model Performance
Linear Regression
Train RMSE: 418,889.63
Train MAE: 270,984.98
Train R²: 0.5732
Test RMSE: 414,957.14
Test MAE: 273,371.74
Test R²: 0.5829
Decision Tree
Train RMSE: 29,340.51
Train MAE: 4,568.66
Train R²: 0.9979
Test RMSE: 65,600.58
Test MAE: 14,839.55
Test R²: 0.9896
Random Forest
Train RMSE: 41,905.39
Train MAE: 5,439.27
Train R²: 0.9957
Test RMSE: 50,953.81
Test MAE: 9,096.14
Test R²: 0.9937
Ridge Regression
Train RMSE: 419,035.95
Train MAE: 271,025.52
Train R²: 0.5729
Test RMSE: 415,288.57
Test MAE: 273,493.66
Test R²: 0.5822
Hyperparameter Tuning Results
Decision Tree
Best Parameters: {'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 40}
Performance: Improved accuracy with better generalization on the test set.
Random Forest
Best Parameters: {'n_estimators': 300, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 20}
Performance: Consistent performance across both training and test sets with reduced error.
Ridge Regression
Best Parameters: {'solver': 'cholesky', 'max_iter': 5000, 'alpha': 1.0}
Performance: Similar to linear regression, with no significant improvement.
Learning Curves and Cross-Validation
Learning Curves: Plots showed that Random Forest exhibited good performance with a stable learning curve, while Decision Tree demonstrated some overfitting.
Cross-Validation: Mean CV Score (Negative MSE): -3,891,131,541.53 with a high standard deviation indicating variability in performance.

Best-Performing Model: 
The Random Forest Regressor with optimized hyperparameters was the best model, achieving:
 RMSE: Lower values compared to other models, indicating fewer prediction errors.
MAE: Lower average error, confirming the model's accuracy in predicting house prices.
R² Score: High R² value, indicating a good fit to the data and strong predictive power.

Key Features:
Building Area: Larger areas were strongly correlated with higher prices.
Distance: Proximity to central locations increased property values.
Total Rooms: More rooms typically indicate higher-priced properties.
Region Name: The location had a significant impact, with some regions consistently showing higher property values.
Feature Importance Visualization: The bar plot of feature importances revealed that `Building Area`, `Distance`, and `Total Rooms` were the most influential features in predicting house prices.
Partial Dependence Plots: Showed that increases in `Building Area` and `Total Rooms` were associated with higher predicted prices, confirming the intuitive understanding of property valuation.


4. Conclusions

Summary: 
The Random Forest model provided the most accurate predictions for house prices, making it the preferred choice for this task. The model's performance metrics indicated that it could effectively capture the complex relationships between features and house prices.
Decision Tree: While it showed excellent training performance, it was prone to overfitting, which led to higher test errors.
Linear Regression and Ridge Regression: Both models performed similarly, with relatively high errors and lower R² values. They were less effective in capturing the underlying data patterns as compared to the more complex models.

Importance of Features: Critical features such as building area, distance to the city centre, and many rooms had the most significant impact on house prices. This aligns with typical market dynamics, where size and location are primary drivers of value.

Implications: The insights from the model can guide buyers and sellers in understanding what drives property values in Melbourne. Real estate agents can leverage these predictions to provide more accurate pricing recommendations.
Future Work: Further improvements could include incorporating additional data sources such as economic indicators or environmental factors. Additionally, experimenting with advanced models like XGBoost or neural networks could further enhance predictive accuracy.

5. Recommendations

Model Choice: Based on performance metrics and generalization capabilities, the Random Forest model is recommended for future use in price forecasting.
Further Improvements: Additional feature engineering, model tuning, and possibly exploring more advanced models or ensemble techniques could further enhance predictive accuracy.
For Buyers: Focus on properties with larger building areas and desirable locations, as these factors significantly influence value.
  
For Sellers: Highlight features like additional rooms or proximity to amenities in listings to justify higher asking prices.
For Real Estate Professionals: Use predictive models as a tool to complement market assessments, allowing for data-driven pricing strategies.


This comprehensive approach not only demonstrated the effectiveness of machine learning in predicting house prices but also provided actionable insights into the real estate market dynamics in Melbourne. The findings and methodologies from this project can be applied to similar datasets in other regions, helping stakeholders make better-informed decisions.

