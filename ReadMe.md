# Stock Market Prediction System

This project is a **Stock Market Prediction System** built using Python and Flask. It integrates LSTM (Long Short-Term Memory) neural networks to analyze stock prices and predict future values. Additionally, it provides real-time stock data and stock information fetched via the `yfinance` API, allowing users to view stock trends and predictions through an interactive web interface.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Route Endpoints](#route-endpoints)
- [Model Training and Prediction](#model-training-and-prediction)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Features

1. **Real-time Stock Data Retrieval**: Fetch current and historical stock data using the `yfinance` API.
2. **LSTM-Based Price Prediction**: Leverages a trained LSTM model to predict stock prices based on past trends.
3. **User Interface**: Intuitive web interface for viewing data, predictions, and insights.
4. **Information Pages**: Includes pages for top gainers, stock information, and contact details.

## Project Structure

Stock Market Prediction System/ 
    ├── final.py         # Main application file 
    ├── templates/  
        ├── index.html 
        ├── stock_data.html 
        ├── finallogin.html 
        ├── finalsignup.html
        └── other_pages... 
    ├── static/           # Static assets (CSS, JS, images)
    ├── README.md         # Project documentation 
    └── requirements.txt  # Python dependencies


## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/stock-market-prediction.git
    cd stock-market-prediction
    ```

2. **Set up a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**:
    ```bash
    python final.py
    ```

    By default, the app runs on `http://127.0.0.1:5000`. You can specify a different port by modifying `app.run(port=8080)`.

### Usage

1. **Start the Flask Server:**
   Run the following command in your terminal to start the server:

   ```bash
   python final.py

-   Once the server is running, open a web browser and go to: `http://127.0.0.1:5000`

-   Go To `Pro Tool` Button. 
-   After Re-direction to the `Prediction page`.
-   Enter the stock in the textbox for Price  Prediction.

### Home Page

- The home page allows users to input a stock symbol and retrieve recent stock data, along with the model’s next-day prediction.


### Stock Data

- Input a stock ticker symbol (e.g., `AAPL` for Apple) on the prediction page to view historical data and predicted future prices.

## Route Endpoints

- `/`: Home page for general information and links to other sections.
- `/predict`: Predicts the next day’s stock price based on the LSTM model.
- `/stock_data`: Displays real-time data and past stock prices.
- `/finallogin.html`: User login page.
- `/finalsignup.html`: User signup page.

## Model Training and Prediction

The LSTM model is trained on 5 years of daily stock data to predict the next day's opening price:

- **Data Normalization**: Scales the input data between 0 and 1 for optimal LSTM performance.
- **LSTM Layers**: Uses a 3-layered LSTM structure with `Dense` output for the final prediction.
- **Training**: The model is trained for 10 epochs with a batch size of 12.
- **Prediction**: Based on past 100-day trends, the model predicts the stock price for the following day.

> **Note**: This is a basic predictive model. For production use, further tuning and evaluation are recommended.

## Technologies Used

- **Python** (Flask, Numpy, Scikit-Learn)
- **Keras** (for building the LSTM model)
- **yFinance** (for fetching stock data)
- **HTML/CSS** (for the web interface)

## Future Enhancements

- **User Profiles**: Allow users to save predictions and monitor multiple stocks.
- **Enhanced Prediction Accuracy**: Experiment with more advanced architectures, like GRUs, and additional financial data.
- **Deployment**: Deploy the app on a cloud platform like Heroku or AWS for wider accessibility.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.