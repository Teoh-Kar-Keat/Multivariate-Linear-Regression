# Technical Report: Predicting Automobile Prices Using Multiple and Ridge Regression Models

## 1.0 Data Exploration and Descriptive Analysis

### 1.1 Dataset Overview

The analysis utilizes a dataset containing various specifications for automobiles to predict their market price. The dependent variable is `Price`, and there are 13 independent variables detailing the physical dimensions, engine characteristics, and performance metrics of the vehicles.

| Variable Type | Variable Name | Description |
| --- | --- | --- |
| Dependent Variable | `Price` | Automobile price |
| Independent Variable | `Wheelbase` | Distance between the centers of front and rear wheels |
| Independent Variable | `Car length` | The length of the automobile |
| Independent Variable | `Car width` | The width of the automobile |
| Independent Variable | `Car height` | The height of the automobile |
| Independent Variable | `Curb weight` | Curb weight of the automobile |
| Independent Variable | `Engine size` | The size of the automobile's engine |
| Independent Variable | `Bore ratio` | The ratio of the cylinder bore to the stroke |
| Independent Variable | `Stroke` | The engine stroke |
| Independent Variable | `Compression ratio` | The engine's compression ratio |
| Independent Variable | `Horsepower` | The horsepower of the engine |
| Independent Variable | `Peak rpm` | The engine's peak revolutions per minute |
| Independent Variable | `City mpg` | Fuel efficiency in miles per gallon for city driving |
| Independent Variable | `Highway mpg` | Fuel efficiency in miles per gallon for highway driving |

*Note: The descriptions for `Curb weight` and `Engine size` have been corrected from the source document, where they were erroneously swapped.*

### 1.2 Analysis of the Dependent Variable: Price

The distribution of the dependent variable, `Price`, was examined to understand its central tendency and spread. The histogram of `Price` reveals a **right-skewed distribution**, indicating that while most car prices cluster at the lower end of the scale, a number of high-value vehicles act as outliers, pulling the mean price upwards. The peak frequency of prices occurs around the $15,000 mark.

### 1.3 Inter-Variable Correlation Analysis

A scatter plot matrix was used to perform an initial visual inspection of the relationships between the independent variables. This analysis revealed several strong correlations, suggesting potential multicollinearity.

- `Wheelbase`, `Car length`, `Car width`, and `Curb weight` show strong positive correlations with each other, which is expected as they are all measures of a vehicle's physical size.
- `Engine size` and `Horsepower` are positively correlated with `Price` and with physical dimension variables like `Curb weight`. This indicates that larger, more powerful cars tend to be heavier and more expensive.
- Fuel efficiency metrics (`City mpg`, `Highway mpg`) are negatively correlated with variables like `Wheelbase`, `Curb weight`, and `Engine size`. This reflects the common trade-off where larger, heavier vehicles typically have lower fuel efficiency.

## 2.0 Initial Model: Multiple Linear Regression

### 2.1 Model Summary and Performance

An initial multiple linear regression model was constructed using all 13 independent variables to predict `Price`. The model demonstrated strong predictive power, explaining a significant portion of the variance in car prices. The **R-squared value was 0.851**, which means the model accounts for approximately 85.1% of the variability in car prices observed in the dataset.

The overall model was found to be statistically significant, as indicated by an **F-statistic of 83.777** with a significance value of **0.000**. This F-test result indicates that the independent variables, taken as a group, have a statistically significant relationship with car price and that the model's predictive capability is not due to random chance.

### 2.2 Diagnosis of Multicollinearity

**Multicollinearity** is a condition in which two or more independent variables in a regression model are highly correlated with one another. When severe multicollinearity is present, the estimation of the regression coefficients can become unstable and unreliable, making it difficult to determine the individual effect of each independent variable on the dependent variable. This instability can manifest in counterintuitive results, such as a variable's coefficient having an illogical sign or an inflated magnitude, which undermines the model's explanatory power. A diagnostic analysis was performed to assess the level of multicollinearity in the initial model.

### 2.2.1 Correlation Matrix

The Pearson correlation matrix of the independent variables confirmed the presence of high linear relationships. Multiple pairs of variables exhibited correlation coefficients greater than 0.7. Notably, the correlation between **`City mpg` and `Highway mpg` was 0.971**, indicating a near-perfect linear relationship that is a significant source of multicollinearity.

### 2.2.2 Variance Inflation Factor (VIF)

The Variance Inflation Factor (VIF) is a metric that quantifies the severity of multicollinearity by measuring how much the variance of an estimated regression coefficient is increased due to collinearity. A common rule of thumb is that a **VIF value greater than 10** indicates a problematic level of multicollinearity. The VIF values for several variables in the model exceeded this threshold, confirming the presence of a severe issue.

| Variable | VIF |
| --- | --- |
| City mpg | 27.129 |
| Highway mpg | 24.277 |
| Curb weight | 16.413 |

These high VIF values confirm that the variance of the coefficient estimates for these variables is highly inflated by their correlations with other predictors, rendering the estimates unreliable.

### 2.2.3 Condition Indices

Further diagnostics, including condition indices greater than 10 and high variance proportions associated with those indices, corroborated the findings from the correlation matrix and VIF analysis. These results collectively provide strong evidence of severe multicollinearity within the initial linear regression model.

## 3.0 Solution: Applying Ridge Regression to Mitigate Multicollinearity

### 3.1 Principles of Ridge Regression

Ridge Regression is a regularization technique specifically designed to address the problem of multicollinearity in linear regression. Its core concept is the addition of a penalty term (known as L2 regularization) to the model's loss function. This penalty is proportional to the square of the magnitude of the coefficients. By penalizing large coefficients, Ridge Regression shrinks them towards zero. This process introduces a small amount of bias into the model but significantly reduces the variance of the coefficient estimates. This deliberate introduction of bias is a tradeoff to achieve a significant reduction in the model's variance. The result is a model that is less sensitive to the idiosyncrasies of the training data and therefore generalizes better to new, unseen data.

### 3.2 Ridge Regression Model Results

To resolve the multicollinearity issue, a Ridge Regression model was applied to the dataset. The table below compares the coefficients from the initial Ordinary Least Squares (OLS) model with the stabilized coefficients produced by the Ridge Regression model.

Comparison of Regression Coefficients | Variable | Initial OLS Coefficient | Ridge Regression Coefficient | | :--- | :--- | :--- | | Wheelbase | 122.617 | 66.473 | | Car length | -94.675 | -21.690 | | Engine size | 117.346 | 84.475 | | Stroke | -3034.606 | -2264.040 | | City mpg | -320.355 | -68.944 | | Highway mpg | 202.822 | -2.992 |

### 3.3 Interpretation of Results

The comparison of coefficients clearly demonstrates the effect of Ridge Regression. The coefficients from the Ridge model are generally smaller in absolute magnitude than those from the initial OLS model, illustrating the shrinkage effect of the L2 penalty. This shrinkage helps to stabilize the model and reduce its sensitivity to the specific training data.

The most dramatic change is observed in the coefficients for `City mpg` and `Highway mpg`. In the initial model, the high correlation between these two variables led to unstable and counterintuitive coefficients, with `Highway mpg` having a large positive coefficient (+202.822). The Ridge model resolved this instability by shrinking both coefficients and correcting the sign of the `Highway mpg` coefficient (from +202.8 to -2.99), which is more aligned with the expected negative relationship between fuel efficiency and price, *all else being equal*. Higher fuel efficiency typically correlates with smaller, less powerful engines, which are features of less expensive cars—a fundamental relationship that was completely obscured by the multicollinearity in the initial OLS model.

## 4.0 Conclusion

A model's predictive power, as measured by R-squared, can be dangerously misleading without rigorous diagnostic testing. This analysis demonstrated that while an initial multiple linear regression model for automobile prices achieved a high R-squared of 0.851, it was fundamentally unreliable due to severe multicollinearity. Diagnostic tools such as the Variance Inflation Factor (VIF) proved essential in uncovering this instability. The application of Ridge Regression did more than correct unstable coefficients; it restored the model's interpretability, ultimately producing a reliable tool for understanding the true drivers of automobile prices. This case underscores a critical principle in predictive modeling: diagnostic diligence is as crucial as predictive accuracy.
