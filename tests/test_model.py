import pytest
import pandas as pd
from src.model import HousePriceModel

def test_model_training():
  data = pd.read_csv('data/house_prices.csv')
  X = data.drop('price',axis=1)
  y = data.data['price']

  model = HousePriceModel()
  mse = model.train(X, y)
  assert mse > 0
