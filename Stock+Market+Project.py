
# coding: utf-8

# # Stock Market Analysis
# 
# Using pandas to get stock information, visualize different aspects of it, and analyze the risk of a stock from previous performance history. Will be using Bootstrap and Monte Carlo Method to predict VaR and stocks
# 
# Info that would be visualized 
# 
# 1. Change in price over time
# 2. Daily return of the stock market average
# 3. Moving average of the various stocks
# 4. Correlation between different stocks' closing price
# 5. Correlation between different stocks' daily returns
# 6. Value that should be put at risk when investing a certain stock
# 7. How to predict future stock behaviour 

# In[4]:


import pandas as pd
from pandas import Series, DataFrame
import numpy as np

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

#For reading stock data 

from pandas_datareader import data, wb
import pandas_datareader as pdr
#For time stamps 
from datetime import datetime 
#Division
from __future__ import division 


# In[5]:


#Using Yahoo to grab stock information

tech_stock = ['GOOG','MSFT','AMZN','AAPL']


# In[6]:


end = datetime.now()
#Setting start and end date (a year ago now)
start = datetime(end.year - 1, end.month, end.day)


# In[7]:


#Setting up Yahoo's financial data as a dataframe 

for stock in tech_stock:
    globals()[stock] = pdr.get_data_yahoo(stock, start,end)


# In[9]:


AAPL.describe() #Statistics on Apple stocks 


# In[10]:


AAPL.info()


# In[11]:


#Historical view on closing price
AAPL['Adj Close'].plot(legend=True, figsize=(10,4))


# In[34]:


#Total volume of stock being traded each day over the course of 5 years
AAPL['Volume'].plot(legend=True,figsize=(12,4))


# In[62]:


#Calculating three moving averages 
#MA = Moving Average
ma_day = [5,10,15,20]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    AAPL[column_name] = Series.rolling(AAPL['Adj Close'],ma).mean()


# In[63]:


AAPL[['Adj Close','MA for 5 days','MA for 10 days','MA for 15 days','MA for 20 days']].plot(subplots=False,figsize=(10,4))


# ## Daily Return Analysis
# 
# - Daily Changes of stocks

# In[13]:


#Percent Change for each day 

AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()
AAPL['Daily Return'].plot(figsize=(16,4),legend=True,linestyle='--',marker='o')


# In[14]:


#Seaborn does not recongize null values so I have to use dropna()
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')


# In[15]:


#Using a histogram plot
AAPL['Daily Return'].hist()


# In[26]:


# Grab all the closing prices for the tech stock list into one DataFrame
closingPrice_dataFrame = pdr.get_data_yahoo(['AAPL','MSFT','AMZN','GOOG'],start,end)['Adj Close']


# In[27]:


closingPrice_dataFrame.head()


# In[28]:


techReturns_dataFrame = closingPrice_dataFrame.pct_change()


# In[29]:


#Comparing Google to Google
sns.jointplot('GOOG','GOOG',techReturns_dataFrame,kind='scatter',color='cyan')


# In[80]:


#Dealing with correlations now 
#Comparing Google with microsoft and seeing if there's any correlation 
sns.jointplot('GOOG','MSFT',techReturns_dataFrame,kind='scatter')


# From what is presented above, there seem to be some sort of correlation forming

# In[81]:


#A sense of what's correlated and what is not
from IPython.display import SVG
SVG(url='http://upload.wikimedia.org/wikipedia/commons/d/d4/Correlation_examples2.svg')


# In[85]:


#To see the correlation for all the companies all at once
sns.pairplot(techReturns_dataFrame.dropna())


# ### Above are all the daily returns between on stocks (correlation) 

# In[87]:


#Full control of the figure, including the upper triangle, and the lower triangle
fig = sns.PairGrid(techReturns_dataFrame.dropna())

fig.map_upper(plt.scatter,color='red')
fig.map_lower(sns.kdeplot,cmap='cool_d')
fig.map_diag(plt.hist,bins=20)


# In[92]:


#Correlation of closing price
fig = sns.PairGrid(closingPrice_dataFrame.dropna())

fig.map_upper(plt.scatter,color='red')
fig.map_lower(sns.kdeplot,cmap='cool_d')
fig.map_diag(plt.hist,bins=20)


# ## Risk Analysis 
# 
# - Gathering daily percentage returns by comparing the expected return with the standard deviation of the daily returns

# In[30]:


returns = techReturns_dataFrame.dropna()


# In[31]:


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


# ## Value at Risk 
# 
# #### Using bootstrap method 
# 
# Calculate the empirical quantiles from a histogram of daily returns. 
# Quantiles information http://en.wikipedia.org/wiki/Quantile

# In[32]:


#Repeating the daily returns for histogram for Apple Stock
#.displot = distribution plot 
sns.distplot(AAPL['Daily Return'].dropna(),bins=100,color='purple')


# In[33]:


#0.05 empirical quantiles of daily returns 
returns['AAPL'].quantile(0.05)


# This is very interesting to see. 
# With 0.05 empirical quantiles, the daily returns is -0.017. 
# The definition of this is that with 95% confidence, the worst daily loss will not surpass 1.7%. 
# If I had \$10 000, my one day VaR (Value at Risk) would be $170

# ## Value at Risk using the Monte Carlo Method 
# 
# Brief Overview: Using the Monte Carlo Method to run many trials with random market conditions and therefore calculate losses/gains for each trial. Then using aggregation to from the trials to establish the risk of a certain stock. 

# The Stock Market will follow a random walk (Markov process) and is following the weak form of EMH (Efficient Market Hypothesis)
# The weak form of EMH states that the next price movement is conditionally dependent on past price movements given that the past prices have been incorporated. 
# 
# This means that the exact price cannot be predicted perfectly solely based on past stocks information
# 
# EMH: https://en.wikipedia.org/wiki/Efficient-market_hypothesis
# 
# Makov Process: https://en.wikipedia.org/wiki/Markov_chain 
# (A stochastic process that satisfies the Markov property if one can make predictions for the future of the process based solely on its present state just as well as one could knowing the process's full history)
# 
# 
# Geometric Browninan Motion equation: (Markov Process)
# ### ΔS/S = μΔt+σϵ√Δt
# In this equation s = stock price, μ = expected return, σ = standard deviation of returns, t = time, ϵ = random variable
# 
# Therefore multiplying the stock price by both sides, the equation is equal to:
# 
# ### ΔS=S(μΔt+σϵ√Δt)
# 
# μΔt is known as the "drift" where average daily returns are multiplied by the change in time. σϵ√Δt is known as the "shock" and the shock is where it'll push the stocks either up or down. By doing drift and shock thousand of times, a simulation can occur to where a stock price might be. 
# Techniques were summarized from here: http://www.investopedia.com/articles/07/montecarlo.asp

# In[35]:


#Setting up the year: 
days = 365

#Setting dt
dt = 1/365

#Finding the dift of Google's dataframe 
mu = returns.mean()['GOOG']

#Calculating the volatility 
sigma = returns.std()['GOOG']


# In[36]:


'''
The following function takes in starting stock prices, 
days of simulations, mu, and sigma. 
It returns a simulated price array 
'''

def stock_monte_carlo(start_price, days, mu, sigma):
    
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in xrange(1,365):
        #calculating the shock (σϵ√Δt )
        shock[x] = np.random.normal(loc=mu*dt,scale = sigma*np.sqrt(dt))
        #calculate Drift
        drift[x] = mu * dt 
        #calculate Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price 


# In[41]:


GOOG.head()


# In[42]:


#Starting price
start_price = 774.50

for i in xrange(365):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))

plt.xlabel("Days")
plt.ylabel("Price")
plt.title("Monte Carlo Analysis for Google")


# In[49]:


#Going to plot the above on a histogram for better visualization 

#Going to run this simulation 10000 times now
runs = 10000
simulations = np.zeros(runs)
np.set_printoptions(threshold = 4) #Or else the output would be far too long to read
for i in xrange(runs):
    #returning [days-1] because we're extracting the end date
    simulations[i] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]; 


# In[50]:


# q as the 1% empirical qunatile therefore 99% of the values should fall between here
q = np.percentile(simulations, 1)
    
# Plotting the distribution of the end prices
plt.hist(simulations,bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.7, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title("Final price distribution for Google Stock after %s days" % days, weight='bold');


# From above, the Value at Risk seems to be \$19.50 for every \$774.50 invested. <br> If a user was putting \$774.50 as an initial investment, it means he's putting \$19.50 at risk. 
