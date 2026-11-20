# Customer Churn Prediction

This project predicts whether a bank customer is likely to leave (churn) or stay. It uses a Custom Artificial Neural Network built with TensorFlow/Keras, and a Streamlit app lets you enter customer details and see the prediction.

`Live Demo Link `: Coming Soon

## Dataset

The model is trained on `Churn_Modelling.csv`, which has 10,000 bank customers. The target column is `Exited` (1 = churned, 0 = stayed).

Features used: Geography, Gender, CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember and EstimatedSalary.

RowNumber, CustomerId and Surname are dropped because they are not useful for prediction. There are no missing values.

## Model

- Gender is label encoded and Geography is one-hot encoded.
- All features are scaled with StandardScaler.
- Data is split 80% for training and 20% for testing.
- The network has 12 inputs, then Dense(64, ReLU), Dense(32, ReLU) and Dense(1, Sigmoid).
- It is trained with the Adam optimizer and binary crossentropy loss, with early stopping on validation loss (patience 10).

Training stopped at epoch 16. Training accuracy was about 87.5% and validation accuracy about 86%.

## Project Files

- `Churn_Modelling.csv` - dataset
- `training.ipynb` - data preparation and model training
- `app.py` - Streamlit app
- `model.h5` - trained model
- `scaler.pkl` - fitted scaler
- `label_encoder_gender.pkl` - fitted gender encoder
- `onehot_encoder_geo.pkl` - fitted geography encoder
- `requirements.txt` - dependencies

## How to Run

Install the dependencies:

    pip install -r requirements.txt

Start the app:

    streamlit run new_app.py


## Retraining

Run all cells in `training.ipynb`. It regenerates the model and the encoder/scaler files. Keep them in the same folder as `app.py`.


