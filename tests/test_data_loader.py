import pytest
from src.data_loader import load_data

def test_load_data():
  data = load_data('data/house_prices.csv')
  assert not data.empty
