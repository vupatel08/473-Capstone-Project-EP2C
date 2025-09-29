from model import Model
from utils.data import load_data

if __name__ == '__main__':
    X, y = load_data()
    m = Model()
    m.fit(X, y)
    print(m.predict(X[:5]))
