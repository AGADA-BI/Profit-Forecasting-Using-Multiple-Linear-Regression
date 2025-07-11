# Company Profit Prediction using Linear Regression

This project implements a linear regression model to predict the profit of 1000 companies based on their operational expenditures and location.

## Project Overview

The goal of this project is to analyze the relationship between various company expenses (R&D spending, administration costs, and marketing spend) along with their geographical location (state) to predict their annual profit using machine learning techniques.

## Dataset

The dataset (`1000_Companies.csv`) contains information about 1000 companies with the following features:

- **R&D Spend**: Research and Development expenditure (in USD)
- **Administration**: Administrative costs (in USD)  
- **Marketing Spend**: Marketing expenditure (in USD)
- **State**: Company location (California, Florida, New York)
- **Profit**: Annual profit (in USD) - *Target variable*

## Project Structure

```
Companies/
├── 1000_Companies.csv                                    # Dataset file
└── Linear Regression to predict the profit of 1000 companies_.ipynb  # Analysis notebook
```

## Analysis Workflow

The Jupyter notebook follows a systematic approach to build and evaluate the prediction model:

### 1. Data Loading and Exploration
- Import necessary libraries (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn)
- Load the dataset and examine its structure
- Display sample data to understand the features

### 2. Data Visualization
- Create a correlation matrix heatmap to visualize relationships between numeric variables
- Analyze feature correlations to understand which factors most influence profit

### 3. Data Preprocessing
- **Feature Engineering**: Convert categorical 'State' variable to numerical format using one-hot encoding
- **Data Splitting**: Divide dataset into training (80%) and testing (20%) sets
- Handle the conversion of boolean dummy variables to integers for model compatibility

### 4. Model Training
- Implement Linear Regression using Scikit-learn
- Train the model on the training dataset
- Extract model coefficients and intercept

### 5. Model Evaluation
- Make predictions on the test set
- Compare actual vs predicted profits
- Calculate R² score to measure model performance
- Analyze model coefficients to understand feature importance

## Key Findings

The linear regression model reveals:
- **Model Coefficients**: Show the impact of each feature on profit prediction
- **R&D Spend**: Has the highest positive coefficient (~0.526), indicating strong correlation with profit
- **Administration**: Second most influential factor (~0.844)
- **Marketing Spend**: Moderate positive impact (~0.108)
- **State Variables**: Location-based adjustments to profit predictions

## Technologies Used

- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Machine learning implementation
- **Jupyter Notebook**: Interactive development environment

## Requirements

To run this project, you'll need:

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## How to Run

1. Clone this repository
2. Install the required dependencies
3. Open the Jupyter notebook: `Linear Regression to predict the profit of 1000 companies_.ipynb`
4. Run all cells sequentially to reproduce the analysis

## Model Performance

The model provides insights into:
- Which business expenses have the greatest impact on profitability
- How geographic location affects company performance
- Prediction accuracy for unseen company data

## Future Improvements

Potential enhancements to this analysis could include:
- Feature scaling/normalization for better model performance
- Cross-validation for more robust model evaluation
- Additional regression techniques (Ridge, Lasso, Polynomial)
- More detailed exploratory data analysis
- Feature importance visualization
- Residual analysis for model validation

## License

This project is open source and available under the MIT License.