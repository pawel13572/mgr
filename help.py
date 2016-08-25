import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("ma_EURUSD.csv", header=0, usecols=[1,7], delimiter=';')
plt.plot(data['Close0'],label="Szereg czasowy")
plt.plot(data['MA120'],label="Srednia kroczaca")
plt.legend()
plt.show()