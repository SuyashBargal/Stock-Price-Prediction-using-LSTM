# Stock Price Prediction using LSTM

## Overview
This project is a Streamlit-based web application that predicts stock prices using a Long Short-Term Memory (LSTM) neural network. The app allows users to select a stock from a dataset and visualize predictions alongside actual stock prices.

## Features
- **Stock Selection:** Users can choose a stock from the sidebar dropdown.
- **Data Preprocessing:** The dataset is loaded, cleaned, and scaled using MinMaxScaler.
- **LSTM Model:** A deep learning model is built using TensorFlow/Keras.
- **Model Training:** Users can trigger model training within the Streamlit app.
- **Evaluation Metrics:** Displays RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
- **Visualization:** Plots actual vs. predicted stock prices.
- **Raw Data Display:** Users can view the stock's historical price data.

## Installation
**Create a Virtual Environment**-
It is recommended to create a virtual environment before installing dependencies to avoid conflicts.
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/stock-prediction-lstm.git
   cd stock-prediction-lstm
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   streamlit run app.py
   ```

## Dependencies
- Python 3.x
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow

## Usage
1. Run the application using Streamlit.
2. Select a stock from the dropdown menu.
3. Click the "Train Model" button to train the LSTM model.
4. View evaluation metrics (RMSE, MAE) and price prediction plots.
5. Optionally, view the raw stock data.

## Dataset
The application loads stock price data from a CSV file (`all_stocks_5yr.csv`). Ensure the dataset is available in the specified path.

## Results
The model was trained on historical stock data, and the results were evaluated using RMSE and MAE. Below are the results for predicting Apple Inc. (AAPL) stock prices:

- **RMSE:** 4.52
- **MAE:** 3.14

## License
This project is open-source and available under the [MIT License](LICENSE).

## Author
For any queries, reach out to suyashbargal@gmail.com

