from fastapi import FastAPI
import tensorflow as tf
from pydantic import BaseModel

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

#buat basemodel untuk post
class DataInput(BaseModel):
    data: str

#import model h5 disini, sebagai contoh inputan model linear
model = tf.keras.models.load_model('./recommender.h5')
#Import the data set
df = pd.read_csv('./Reviews3.csv')

#Dropping the columns
df = df.drop(['Id','HelpfulnessNumerator','HelpfulnessDenominator'], axis = 1) 


@app.get("/")
def hello():
    return {"message": "MODEL API"}

#contoh untuk get model predict
@app.get("/predict")
def predict():
    data = 'A2MUGFV2TDQ47K'
    if data in df['UserId'].values and df['UserId'].size < 5:
        # Top 10 based on rating
        most_rated = df.groupby('ProductId').size().sort_values(ascending=False)[:10]
        final_result = most_rated.index.tolist()
    else:
        counts = df['UserId'].value_counts()
        df_final = df[df['UserId'].isin(counts[counts >= 40].index)]

        # Create a dictionary mapping user IDs to unique indices
        user_ids = df_final['UserId'].unique()
        user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
        index_to_user_id = {index: user_id for index, user_id in enumerate(user_ids)}

        # Create a dictionary mapping product IDs to unique indices
        product_ids = df_final['ProductId'].unique()
        product_id_to_index = {product_id: index for index, product_id in enumerate(product_ids)}
        index_to_product_id = {index: product_id for index, product_id in enumerate(product_ids)}

        # Convert user and product IDs to indices in the dataframe
        df_final['user_index'] = df_final['UserId'].map(user_id_to_index)
        df_final['product_index'] = df_final['ProductId'].map(product_id_to_index)

        uidx = df_final['user_index'].values.astype(np.int64)
        pidx = df_final['product_index'].values.astype(np.int64)

        # Create a new DataFrame with converted data arrays
        df_converted = pd.DataFrame({'UserId': uidx, 'ProductId': pidx, 'Score': 0})

        # Create pivot table with the converted DataFrame
        final_ratings_matrix = pd.pivot_table(df_converted, index='UserId', columns='ProductId', values='Score')
        final_ratings_matrix.fillna(0, inplace=True)

        array3 = final_ratings_matrix.reset_index().melt(id_vars=['UserId'], value_vars=final_ratings_matrix.columns)
        array3 = array3[['UserId', 'ProductId']].values.astype(np.int64)

        # Filter the array3 for the specific user ID
        filtered_array3 = array3[array3[:, 0] == user_id_to_index[data]]

        # Perform predictions
        predictions = model.predict(filtered_array3)

        # Inverse transform the scaled ratings to get the actual ratings
        scaler = MinMaxScaler()
        score = scaler.fit_transform(df['Score'].values.reshape(-1, 1))
        predictions = scaler.inverse_transform(predictions)

        # Make prediction result to df
        df_predicted = pd.DataFrame(filtered_array3, columns=['UserId', 'ProductId'])
        df_predicted['PredictedRatings'] = predictions
        df_predicted = df_predicted.sort_values(by='PredictedRatings', ascending=False)

        # Rename the columns back to 'UserId' and 'ProductId'
        df_predicted = df_predicted.rename(columns={'UserId': 'user_index', 'ProductId': 'product_index'})

        # Convert the user index back to 'UserId' and product index back to 'ProductId'
        df_predicted['UserId'] = df_predicted['user_index'].map(index_to_user_id)
        df_predicted['ProductId'] = df_predicted['product_index'].map(index_to_product_id)

        final_result = df_predicted[['UserId', 'ProductId', 'PredictedRatings']].head(10).values.tolist()

        # Convert predictions to a JSON response
        response = {'predictions': final_result}
    return response


#contoh untuk post model predict
@app.post("/predict")
def predict(data: DataInput):
    data = data.data
    if data in df['UserId'].values and df['UserId'].size < 5:
        # Top 10 based on rating
        most_rated = df.groupby('ProductId').size().sort_values(ascending=False)[:10]
        final_result = most_rated.index.tolist()
    else:
        counts = df['UserId'].value_counts()
        df_final = df[df['UserId'].isin(counts[counts >= 40].index)]

        # Create a dictionary mapping user IDs to unique indices
        user_ids = df_final['UserId'].unique()
        user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
        index_to_user_id = {index: user_id for index, user_id in enumerate(user_ids)}

        # Create a dictionary mapping product IDs to unique indices
        product_ids = df_final['ProductId'].unique()
        product_id_to_index = {product_id: index for index, product_id in enumerate(product_ids)}
        index_to_product_id = {index: product_id for index, product_id in enumerate(product_ids)}

        # Convert user and product IDs to indices in the dataframe
        df_final['user_index'] = df_final['UserId'].map(user_id_to_index)
        df_final['product_index'] = df_final['ProductId'].map(product_id_to_index)

        uidx = df_final['user_index'].values.astype(np.int64)
        pidx = df_final['product_index'].values.astype(np.int64)

        # Create a new DataFrame with converted data arrays
        df_converted = pd.DataFrame({'UserId': uidx, 'ProductId': pidx, 'Score': 0})

        # Create pivot table with the converted DataFrame
        final_ratings_matrix = pd.pivot_table(df_converted, index='UserId', columns='ProductId', values='Score')
        final_ratings_matrix.fillna(0, inplace=True)

        array3 = final_ratings_matrix.reset_index().melt(id_vars=['UserId'], value_vars=final_ratings_matrix.columns)
        array3 = array3[['UserId', 'ProductId']].values.astype(np.int64)

        # Filter the array3 for the specific user ID
        filtered_array3 = array3[array3[:, 0] == user_id_to_index[data]]

        # Perform predictions
        predictions = model.predict(filtered_array3)

        # Inverse transform the scaled ratings to get the actual ratings
        scaler = MinMaxScaler()
        score = scaler.fit_transform(df['Score'].values.reshape(-1, 1))
        predictions = scaler.inverse_transform(predictions)

        # Make prediction result to df
        df_predicted = pd.DataFrame(filtered_array3, columns=['UserId', 'ProductId'])
        df_predicted['PredictedRatings'] = predictions
        df_predicted = df_predicted.sort_values(by='PredictedRatings', ascending=False)

        # Rename the columns back to 'UserId' and 'ProductId'
        df_predicted = df_predicted.rename(columns={'UserId': 'user_index', 'ProductId': 'product_index'})

        # Convert the user index back to 'UserId' and product index back to 'ProductId'
        df_predicted['UserId'] = df_predicted['user_index'].map(index_to_user_id)
        df_predicted['ProductId'] = df_predicted['product_index'].map(index_to_product_id)

        final_result = df_predicted[['UserId', 'ProductId', 'PredictedRatings']].head(10).values.tolist()

        # Convert predictions to a JSON response
        response = {'predictions': final_result}
    return response




