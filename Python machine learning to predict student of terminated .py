import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel('.....')

df


plt.scatter(x='x',y='y',data=df)
plt.grid()
plt.show()

x=df.x.values.reshape(-1,1)
y=df.y.values.reshape(-1,1)

x,y


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_feature=PolynomialFeatures(degree=1)
x_poly=poly_feature.fit_transform(x)
model=LinearRegression()
model.fit(x_poly,y)

y_poly_pred=model.predict(x_poly)
sns.set_style('darkgrid')
plt.scatter(x,y,color='b',s=20)
plt.plot(x,y_poly_pred,color='r')
plt.ylabel('Batch student')
plt.xlabel('student terminated')
plt.title('student terminated')
plt.show()

from sklearn.metrics import r2_score

print('R2={:.5f}'.format(r2_score(y,y_poly_pred)))

x_input=[[15],[16],[17]]

y_poly_pred=model.predict(poly_feature.fit_transform(x_input))
y_poly_pred

for val in y_poly_pred:
    print('{:.0f}'.format(val))


