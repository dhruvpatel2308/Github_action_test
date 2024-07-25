from sklearn.liner_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class HousePriceModel:
  def __init__(self):
    self.model = LinearRegression()

  def train(self, X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    self.model.fit(X_train, y_train)
    predictions = self.model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def predict(self, X):
  return self.model.predict(X)
