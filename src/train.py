import pandas as pd
from data_loader import load_data
from model import HousePriceModel

def main():
  data = load_data('data/house_prices.csv)
  X = data.drop('price',axis=1)
  y = data['price']

  model = HousePriceModel()
  mse = model.train(X,y)
  print(f"Model trained with MSE: {mse}")

if __name__ == '__main__':
  main()
