# Admission Prediction using Linear Regression

This project uses linear regression to predict the chance of admission based on GRE scores. The dataset is loaded, split into training and testing sets, and a linear regression model is applied to understand the relationship between GRE Score and the Chance of Admission.

## Dataset
- The dataset used is `Admission_Predict.csv`.
- It contains multiple features, but only the GRE Score is used as the predictor in this model.

## Requirements
Install the required libraries using:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Code Explanation
1. **Import Libraries:**
   - `numpy`, `pandas` for data manipulation.
   - `matplotlib` for plotting.
   - `sklearn` for machine learning models and evaluation.

2. **Load Dataset:**
   ```python
   dataset = pd.read_csv('/Admission_Predict.csv')
   ```

3. **Data Preprocessing:**
   - Extract the GRE Score as the feature (`X`) and Chance of Admission as the target (`Y`).
   - Split the dataset into training and testing sets (80% train, 20% test).

4. **Train the Model:**
   - Use `LinearRegression` from `sklearn` to train the model.
   ```python
   regressor = LinearRegression()
   regressor.fit(X_train, Y_train)
   ```

5. **Prediction and Visualization:**
   - Predict on the test set and plot training and test results.
   - Scatter plots show actual data points, and the regression line shows predictions.

6. **Performance Evaluation:**
   - Calculate Residual Sum of Squares (RSS) and Residual Standard Error (RSE).
   ```python
   RSS = sum((Y_test - y_pred) ** 2)
   RSE = sqrt(RSS / (len(Y_test) - 2))
   print("RSS IS: ", RSS)
   print("RSE IS: ", RSE)
   ```

## Results
- The model shows the relationship between GRE Score and the Chance of Admission.
- Performance metrics like RSS and RSE help evaluate the model's accuracy.

## Usage
1. Place the `Admission_Predict.csv` file in the appropriate directory.
2. Run the script to train the model and visualize the results.

## Visualization
Two scatter plots will be generated:
- Training Set: Shows the model's fit on training data.
- Test Set: Compares actual test data with predicted values.

## Conclusion
This project demonstrates a simple linear regression model for predicting admission chances based on GRE scores. Further improvements can include using multiple features and testing other regression techniques.

---


