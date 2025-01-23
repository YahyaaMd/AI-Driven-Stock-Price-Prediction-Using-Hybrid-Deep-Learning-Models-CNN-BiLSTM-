

# AI-Driven Stock Price Prediction Using Hybrid Deep Learning Models (CNN-BiLSTM)

## Overview
This project aims to predict stock prices using a hybrid deep learning model combining Convolutional Neural Networks (CNN) and Bidirectional Long Short-Term Memory (BiLSTM) networks. By leveraging historical stock price data from Yahoo Finance, the model provides more accurate predictions by capturing both spatial and temporal dependencies in the data.

## Dataset
The datasets used in this project are historical stock prices for:
- **Google (GOOG)**
- **Microsoft (MSFT)**

These datasets were obtained from [Yahoo Finance](https://finance.yahoo.com/) and include the following columns:
- **Date**: The date of the stock price.
- **Open**: Opening price of the stock.
- **High**: Highest price of the stock on the given day.
- **Low**: Lowest price of the stock on the given day.
- **Close**: Closing price of the stock on the given day.
- **Adj Close**: Adjusted closing price, accounting for splits and dividends.
- **Volume**: Number of shares traded.

## Methodology
1. **Data Preprocessing**
   - Loaded and cleaned the datasets to handle missing values and ensure consistency.
   - Normalized the data to improve model performance.
   - Split the data into training and testing sets.

2. **Feature Engineering**
   - Selected relevant features for model input (e.g., Open, High, Low, Close).
   - Created sequences of historical data points for temporal analysis.

3. **Model Architecture**
   - **CNN**: Extracts spatial features from input data.
   - **BiLSTM**: Captures temporal dependencies and bidirectional trends.
   - Combined these models into a hybrid architecture to leverage the strengths of both approaches.

4. **Training and Evaluation**
   - Trained the model on historical data using Mean Squared Error (MSE) as the loss function.
   - Evaluated the model using metrics like Root Mean Squared Error (RMSE).
   - Visualized the predicted vs. actual stock prices to assess model accuracy.

## Requirements
To run this project, you'll need the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tensorflow`
- `scikit-learn`

Install the required libraries using:
```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Driven-Stock-Prediction
```

2. Navigate to the project directory:
```bash
cd AI-Driven-Stock-Prediction
```

3. Place the datasets (`GOOG.csv` and `MSFT.csv`) in the `data` folder.

4. Run the Jupyter Notebook:
```bash
jupyter notebook "AI Driven Stock Price Prediction Using Hybrid Deep Learning Models (CNN-BiLSTM).ipynb"
```

5. Follow the steps in the notebook to preprocess data, train the model, and visualize predictions.

## Results
The hybrid CNN-BiLSTM model demonstrated:
- Improved prediction accuracy by effectively capturing spatial and temporal features.
- Visualized performance through line plots comparing predicted and actual stock prices.

## Contributions
Feel free to contribute to this project by:
- Adding new features or datasets.
- Experimenting with other hybrid model architectures.
- Improving the preprocessing pipeline or model performance.

## License
This project is licensed under the MIT License.

## Acknowledgements
- Data sourced from [Yahoo Finance](https://finance.yahoo.com/).
- Developed using Python and TensorFlow.


