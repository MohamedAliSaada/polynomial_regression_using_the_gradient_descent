# Polynomial Regression with Gradient Descent

This repository contains a Python implementation of a polynomial regression model using gradient descent. The goal is to predict `Y` from `X` based on a polynomial function, and the implementation includes data preprocessing, normalization, cost function evaluation, and convergence analysis.

## Code Explanation

1. **Data Preparation**:
   - The data is read from a CSV file (`ex_3.csv`), where `X` and `Y` values are extracted. 
   - `X` is transformed into polynomial features up to the 4th degree (i.e., `x`, `x²`, `x³`, and `x⁴`).
   - The features are then normalized to have zero mean and unit variance using standardization.

2. **Hyperparameters**:
   - The learning rate (`alpha`) is set to 0.03, and the number of epochs is set to 1000. These are key parameters that control the training process.

3. **Gradient Descent**:
   - The gradient descent algorithm is applied to minimize the cost function. At each epoch, the predicted values are calculated and compared to the actual values (`Y`).
   - The gradients with respect to the parameters (weights and bias) are computed, and the weights and bias are updated using the learning rate.

4. **Cost Function**:
   - A cost function is used to measure the performance of the model. The goal of gradient descent is to minimize this cost. The cost is computed as the sum of squared errors between the predicted and actual values.

5. **Convergence Check**:
   - The script tracks the cost function history over epochs to ensure that the model is converging. A graph is generated to visualize the convergence of the cost function.

   **Convergence of the Cost Function**:

   ![Convergence](https://github.com/MohamedAliSaada/polynomial_regression_using_the_gradient_descent/blob/main/convergence.png)

   - The plot shows the cost history, with the value dropping significantly as the number of epochs increases, indicating that the model is learning.
   - The point marked with a blue dot at epoch 999 represents the minimum cost of 194.45, showing where the algorithm converged.

6. **Final Model**:
   - After training, the weights and bias are transformed back to the original scale of the features using reverse scaling. The final polynomial equation is printed, showing the relationship between `Y` and `X` in its real scale.
   ![Module](https://github.com/MohamedAliSaada/polynomial_regression_using_the_gradient_descent/blob/main/Module_results.png)
   Example output of the polynomial equation:
   ```python
   Real scale polynomial equation: y_pred_ = 123.45 + 2.56x + 3.78x^2 + 4.90x^3 + 5.67x^4
