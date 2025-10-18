# Tic-Tac-Toe AdaBoost Implementation

A machine learning project implementing the AdaBoost algorithm for binary classification using the classic Tic-Tac-Toe endgame dataset.

## Overview

This project demonstrates the implementation of the AdaBoost (Adaptive Boosting) algorithm to classify tic-tac-toe board positions as winning or losing scenarios. The implementation uses decision stumps (single-level decision trees) as weak learners and combines them to create a strong classifier.

## Dataset

The project uses the **Tic-Tac-Toe Endgame Database** from the UCI Machine Learning Repository:

- **Source**: David W. Aha (aha@cs.jhu.edu)
- **Date**: August 19, 1991
- **Description**: Contains all possible board configurations for tic-tac-toe endgame positions
- **Features**: 9 attributes representing the 9 squares of a tic-tac-toe board
- **Classes**: 
  - `positive`: Winning positions for X
  - `negative`: Non-winning positions for X

## Features

- **Custom AdaBoost Implementation**: Built from scratch using NumPy and scikit-learn components
- **5-Fold Cross-Validation**: Ensures robust model evaluation
- **Visualization**: Comprehensive plotting of training and validation metrics
- **Data Preprocessing**: Automatic encoding of categorical features
- **Performance Metrics**: Accuracy and error tracking across iterations

## Project Structure

```
TicTacToe-AdaBoost-Implementation/
├── main.ipynb                     # Main Jupyter notebook with implementation
├── data/
│   ├── Tic-Tac-Toe Endgame.txt   # Dataset documentation
│   ├── tttoe.csv                 # Original dataset
│   └── tttoe_encoded.csv         # Preprocessed dataset
└── README.md                     # This file
```

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib

## Installation

1. Clone this repository:
```bash
git clone https://github.com/waisman-braga/TicTacToe-AdaBoost-Implementation.git
cd TicTacToe-AdaBoost-Implementation
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

2. Run all cells to:
   - Load and preprocess the tic-tac-toe dataset
   - Train the AdaBoost model with 300 estimators
   - Perform 5-fold cross-validation
   - Generate accuracy and error plots

## Algorithm Details

### AdaBoost Implementation

The AdaBoost algorithm works by:

1. **Initialization**: Setting equal weights for all training samples
2. **Iterative Training**: 
   - Training a decision stump on weighted data
   - Calculating the stump's error rate
   - Computing the stump's weight based on its performance
   - Updating sample weights (increasing weights for misclassified samples)
3. **Final Prediction**: Combining all stumps using weighted voting

### Key Parameters

- **n_estimators**: 300 (number of boosting iterations)
- **max_depth**: 1 (decision stumps)
- **random_state**: 42 (for reproducibility)

## Results

The implementation achieves:

- **Training Accuracy**: ~98% after 200 estimators
- **Validation Accuracy**: Consistent performance across folds
- **Convergence**: Model stabilizes around estimator 200
- **Error Reduction**: Linear decrease in error rate over iterations

## Visualization

The project generates several plots:

1. **Individual Fold Performance**: Training vs validation accuracy for each cross-validation fold
2. **Mean Performance**: Average training and validation accuracy across all folds
3. **Error Analysis**: Training and validation error trends over iterations

## Key Insights

- The model shows excellent learning capability with increasing estimators
- After 200 iterations, the model reaches near-optimal performance (~98% accuracy)
- The error exhibits a consistent linear decrease, indicating effective boosting
- Cross-validation ensures the results are robust and generalizable

## Contributing

Feel free to fork this repository and submit pull requests for improvements or additional features.

## License

This project is open source and available under the [MIT License](LICENSE).

## References

1. Matheus, C. J., & Rendell, L. A. (1989). Constructive induction on decision trees. In Proceedings of the Eleventh International Joint Conference on Artificial Intelligence (pp. 645--650).

2. UCI Machine Learning Repository: Tic-Tac-Toe Endgame Data Set