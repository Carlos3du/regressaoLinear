# Boston Housing Regression Project

This project involves using the Boston Housing dataset to predict the median value of owner-occupied homes (`MEDV`) using regression techniques. The primary focus is on utilizing the K-Nearest Neighbors (KNN) algorithm and evaluating its performance based on various metrics.

## Project Overview

The Boston Housing dataset contains information about different factors that influence housing prices in Boston. The goal is to build a regression model that accurately predicts housing prices (`MEDV`) based on the features provided.

### Key Steps in the Project:
1. **Data Preprocessing**:
   - Load the dataset and clean it by handling missing or null values.
   - Normalize the features using `MinMaxScaler` for better model performance.
   - Remove outliers using the Interquartile Range (IQR) method.

2. **Exploratory Data Analysis (EDA)**:
   - Visualize the distribution of the target variable (`MEDV`).
   - Analyze the correlation between features and the target variable.
   - Generate pair plots and box plots for high-correlation features.

3. **Feature Selection**:
   - Select features with high correlation to `MEDV` for model training.

4. **Model Training and Evaluation**:
   - Split the data into training and testing sets.
   - Train the KNN regressor with different values of `K`.
   - Evaluate the model using metrics such as:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Mean Absolute Percentage Error (MAPE)
   - Select the best `K` value based on the evaluation metrics.

5. **Visualization of Results**:
   - Plot the regression results to compare predicted vs. actual values.

## Dataset Description

The dataset contains the following features:

| Feature   | Description                                                                 |
|-----------|-----------------------------------------------------------------------------|
| CRIM      | Per capita crime rate by town                                              |
| ZN        | Proportion of residential land zoned for lots over 25,000 sq.ft.           |
| INDUS     | Proportion of non-retail business acres per town                           |
| CHAS      | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)      |
| NOX       | Nitric oxides concentration (parts per 10 million)                         |
| RM        | Average number of rooms per dwelling                                       |
| AGE       | Proportion of owner-occupied units built prior to 1940                     |
| DIS       | Weighted distances to five Boston employment centers                       |
| RAD       | Index of accessibility to radial highways                                  |
| TAX       | Full-value property-tax rate per $10,000                                   |
| PTRATIO   | Pupil-teacher ratio by town                                                |
| B         | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town             |
| LSTAT     | % lower status of the population                                           |
| MEDV      | Median value of owner-occupied homes in $1000's (Target Variable)          |

## Requirements

The following Python libraries are required to run the project:
- `pandas`
- `numpy`
- `scikit-learn`
- `seaborn`
- `matplotlib`

## How to Run

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Open the `notebook.ipynb` file in Jupyter Notebook or any compatible environment.
4. Run the cells sequentially to execute the project.

## Results

The project identifies the optimal value of `K` for the KNN regressor and evaluates its performance using the metrics mentioned above. Visualizations are provided to analyze the results and understand the model's predictions.

## License

This project is licensed under the MIT License.