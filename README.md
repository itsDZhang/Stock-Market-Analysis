
# Stock Market Analysis

Using pandas to get stock information, visualize different aspects of it, and analyze the risk of a stock from previous performance history. Then predicting using Monte Carlo Method

Info that would be visualized 

1. Change in price over time
2. Daily return of the stock market average
3. Moving average of the various stocks
4. Correlation between different stocks' closing price
5. Correlation between different stocks' daily returns
6. Value that should be put at risk when investing a certain stock
7. How to predict future stock behaviour 


```python
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

#For reading stock data 

from pandas_datareader import data, wb
import pandas_datareader as pdr
#For time stamps 
from datetime import datetime 
#Division
from __future__ import division 
```


```python
#Using Yahoo to grab stock information

tech_stock = ['GOOG','MSFT','AMZN','AAPL']
```


```python
end = datetime.now()
#Setting start and end date (a year ago now)
start = datetime(end.year - 1, end.month, end.day)
```


```python
#Setting up Yahoo's financial data as a dataframe 

for stock in tech_stock:
    globals()[stock] = pdr.get_data_yahoo(stock, start,end)
```


```python
AAPL.describe() #Statistics on Apple stocks 
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Adj Close</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>251.000000</td>
      <td>251.000000</td>
      <td>251.000000</td>
      <td>251.000000</td>
      <td>251.000000</td>
      <td>2.510000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>141.341315</td>
      <td>142.295060</td>
      <td>140.435937</td>
      <td>141.474103</td>
      <td>140.677993</td>
      <td>2.793747e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>16.902716</td>
      <td>16.923112</td>
      <td>16.647013</td>
      <td>16.736888</td>
      <td>17.202225</td>
      <td>1.190328e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>106.570000</td>
      <td>107.680000</td>
      <td>104.080002</td>
      <td>105.709999</td>
      <td>104.410980</td>
      <td>1.147590e+07</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>130.944999</td>
      <td>132.154998</td>
      <td>130.784996</td>
      <td>131.784996</td>
      <td>130.165566</td>
      <td>2.067555e+07</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>144.449997</td>
      <td>145.300003</td>
      <td>143.449997</td>
      <td>144.289993</td>
      <td>143.610962</td>
      <td>2.533170e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>154.799995</td>
      <td>155.470001</td>
      <td>153.805001</td>
      <td>154.735001</td>
      <td>154.431870</td>
      <td>3.221495e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>174.000000</td>
      <td>174.259995</td>
      <td>171.119995</td>
      <td>172.500000</td>
      <td>172.500000</td>
      <td>1.119850e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
AAPL.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 251 entries, 2016-11-07 to 2017-11-03
    Data columns (total 6 columns):
    Open         251 non-null float64
    High         251 non-null float64
    Low          251 non-null float64
    Close        251 non-null float64
    Adj Close    251 non-null float64
    Volume       251 non-null int64
    dtypes: float64(5), int64(1)
    memory usage: 13.7 KB



```python
#Historical view on closing price
AAPL['Adj Close'].plot(legend=True, figsize=(10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1070d3a10>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_7_1.png?raw=true)



```python
#Total volume of stock being traded each day over the course of 5 years
AAPL['Volume'].plot(legend=True,figsize=(12,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a10af4090>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_8_1.png?raw=true)



```python
#Calculating three moving averages 
#MA = Moving Average
ma_day = [5,10,15,20]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    AAPL[column_name] = Series.rolling(AAPL['Adj Close'],ma).mean()

```


```python
AAPL[['Adj Close','MA for 5 days','MA for 10 days','MA for 15 days','MA for 20 days']].plot(subplots=False,figsize=(10,4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a119b6710>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_10_1.png?raw=true)


## Daily Return Analysis

- Daily Changes of stocks


```python
#Percent Change for each day 

AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(16,4),legend=True,linestyle='--',marker='o')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a1214ca10>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_12_1.png?raw=true)



```python
#Seaborn does not recongize null values so I have to use dropna()
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a11dd53d0>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_13_1.png?raw=true)



```python
#Using a histogram plot
AAPL['Daily Return'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1a12630bd0>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_14_1.png?raw=true)



```python
# Grab all the closing prices for the tech stock list into one DataFrame
closingPrice_dataFrame = pdr.get_data_yahoo(['AAPL','MSFT','AMZN','GOOG'],start,end)['Adj Close']
```


```python
closingPrice_dataFrame.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AAPL</th>
      <th>AMZN</th>
      <th>GOOG</th>
      <th>MSFT</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-11-03</th>
      <td>172.500000</td>
      <td>1111.599976</td>
      <td>1032.479980</td>
      <td>84.139999</td>
    </tr>
    <tr>
      <th>2017-11-02</th>
      <td>168.110001</td>
      <td>1094.219971</td>
      <td>1025.579956</td>
      <td>84.050003</td>
    </tr>
    <tr>
      <th>2017-11-01</th>
      <td>166.889999</td>
      <td>1103.680054</td>
      <td>1025.500000</td>
      <td>83.180000</td>
    </tr>
    <tr>
      <th>2017-10-31</th>
      <td>169.039993</td>
      <td>1105.280029</td>
      <td>1016.640015</td>
      <td>83.180000</td>
    </tr>
    <tr>
      <th>2017-10-30</th>
      <td>166.720001</td>
      <td>1110.849976</td>
      <td>1017.109985</td>
      <td>83.889999</td>
    </tr>
  </tbody>
</table>
</div>




```python
techReturns_dataFrame = closingPrice_dataFrame.pct_change()
```


```python
#Comparing Google to Google
sns.jointplot('GOOG','GOOG',techReturns_dataFrame,kind='scatter',color='cyan')
```




    <seaborn.axisgrid.JointGrid at 0x1a12968410>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_18_1.png?raw=true)



```python
#Dealing with correlations now 
#Comparing Google with microsoft and seeing if there's any correlation 
sns.jointplot('GOOG','MSFT',techReturns_dataFrame,kind='scatter')
```




    <seaborn.axisgrid.JointGrid at 0x1a12c35f90>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_19_1.png?raw=true)


From what is presented above, there seem to be some sort of correlation forming


```python
#A sense of what's correlated and what is not
from IPython.display import SVG
SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')
```




![svg](https://i.gyazo.com/962eccac8e3b1f398eaadcc6852d2e18.png)




```python
#To see the correlation for all the companies all at once
sns.pairplot(techReturns_dataFrame.dropna())
```




    <seaborn.axisgrid.PairGrid at 0x1a13cfb150>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_22_1.png?raw=true)


### Above are all the daily returns between on stocks (correlation) 


```python
#Full control of the figure, including the upper triangle, and the lower triangle
fig = sns.PairGrid(techReturns_dataFrame.dropna())

fig.map_upper(plt.scatter,color='red')
fig.map_lower(sns.kdeplot,cmap='cool_d')
fig.map_diag(plt.hist,bins=20)
```




    <seaborn.axisgrid.PairGrid at 0x1a13d13890>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_24_1.png?raw=true)



```python
#Correlation of closing price
fig = sns.PairGrid(closingPrice_dataFrame.dropna())

fig.map_upper(plt.scatter,color='red')
fig.map_lower(sns.kdeplot,cmap='cool_d')
fig.map_diag(plt.hist,bins=20)
```




    <seaborn.axisgrid.PairGrid at 0x1a13bb1610>




![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_25_1.png?raw=true)


## Risk Analysis 

- Gathering daily percentage returns by comparing the expected return with the standard deviation of the daily returns


```python
returns = techReturns_dataFrame.dropna()
```


```python
plt.scatter(returns.mean(), returns.std(), alpha=0.5,s = np.pi*20)
#Setting limits
plt.ylim([0.005,0.025])
plt.xlim([-0.003,0.002])

#Set the plot axis titles
plt.xlabel('Expected Returns')
plt.ylabel('Risk')

#Labelling the scatterplots:  http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(returns.columns, returns.mean(), returns.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))
```


![png](https://github.com/DZcoderX/Stock-Market-Analysis/blob/master/Graphs/output_28_0.png?raw=true)

