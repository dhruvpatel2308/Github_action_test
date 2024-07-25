import pandas as pd
from data_loader import load_data
from model import HousePriceModel

def main():
  model = HousePriceModel()
  model.model.load('saved_model')

  data = load_data('path_to_data.csv')

  predictions=model.predict(data)
  print(predictions)
if __name__ == '__main__':
  main()
