# Name - Abhay Gupta
# registration no. - B20075
# Mobile - 9511334630

#%%
#import libraries
from os import sep
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg as AR
import math
from sklearn.metrics import mean_squared_error
import matplotlib

#use pandas to read csv file
covid_data = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep = ",")



#### Question 1 ####
print("Question 1:-")
# (a) part
# Creating the line plot between date and number of COVID-19 new cases

df = pd.read_csv('daily_covid_cases.csv',sep=",")
plt.plot(df["new_cases"])
plt.title("Number of Covid-19 cases")

xt = np.array(df['Date'])
for i in range(xt.shape[0]):
 xt[i] = xt[i][3:]
plt.xticks(np.arange(0,df.shape[0],100),labels=xt[::100])
plt.show()

# (b) part
# making the function autocor which returns the autocorrelation value 
# a is the time series and p is the time lag value
def autocorr(a,p):
    a1 = a.iloc[p:]
    a2 = a.iloc[:-p]
    cor = np.corrcoef(a1,a2)[0][1]
    return cor

#finding the correlation coefficient between the original series and the time lag series
corr1b = autocorr(covid_data['new_cases'],1)
print("Pearson correlation (autocorrelation) coefficient between the generated one-day lag time sequence and the given time sequence: ",corr1b)

# (c) part
# making a function to plot the scatter plot between original time sequence and one day lag time sequence
def pltscattr(a,b,xl,yl):
    # a,b are the original time sequence and the time lag sequence respectively
    plt.figure()
    plt.scatter(a,b)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.show()

pltscattr(covid_data['new_cases'].iloc[1:],covid_data['new_cases'].iloc[:-1],"Original time sequence","One day lag sequence")

# (d) part
#initialising the array for storing autocorrelation values for different values of p
autocor_values = [0]*6

# making an array of lagged values
lagged_values = [1,2,3,4,5,6]

# finding the autocorrelation values using autocorrelation function which was defined above
for i in range(6):
    autocor_values[i] = autocorr(covid_data['new_cases'],i+1)

# Plotting the a line plot between obtained correlation coefficients (on the y-axis) and lagged values (on the x-axis).
plt.figure()
plt.plot(lagged_values,autocor_values,scaley=True)
plt.xlabel('Lagged Values')
plt.ylabel('Correlation coefficients')
plt.show()

# (e) part
# Plotting a correlogram or Auto Correlation Function 
sm.graphics.tsa.plot_acf((covid_data['new_cases']))
plt.show()


#### Question 2 ####
print("\n Question 2 :-")
# (a) part
# splitting the data into training and test data
test_size = 0.35  # 35% for testing
X = covid_data.values
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# plotting the training and test sets
plt.figure()
plt.plot(train)
plt.title("Training data")
plt.show()

plt.figure()
plt.plot(test)
plt.title("Test data")
plt.show()

# making the auto regression model which takes inout as the training data, test data and the time lag value 'p'
coefficients = []
model = AR(train, lags=5) 
# fit/train the model
model_fit = model.fit() 
# Get the coefficients of AR model
coef = model_fit.params 
for i in coef:
    coefficients.append(i)

def autoreg_model(train,test,p):
    model = AR(train, lags=p) 
    # fit/train the model
    model_fit = model.fit() 
    # Get the coefficients of AR model
    coef = model_fit.params 
    

    # using these coefficients walk forward over time steps in test, one step each time
    history = train[len(train)-p:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-p,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(p):
            yhat += coef[d+1] * lag[p-d-1] # Add other values
        obs = test[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.
    return predictions
window = 5 # The lag=5
# finding the predicted values of the test data
predictions = autoreg_model(train,test,window)

# (b) part

# (i) part
# plotting the scatter plot
plt.figure()
plt.scatter(test,predictions)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()

# (ii) part
plt.figure()
plt.plot(test,predictions)
plt.xlabel("Actual values")
plt.ylabel("Predicted values")
plt.show()


# (iii) part
# making the function which calculates rmse value (in %)
def rmse(X_pred,X):
    e= (mean_squared_error(X, X_pred)**0.5) 
    s = 0
    for i in range(len(X)):
        s += X[i]
    m = s/len(X)
    error = e*100/m
    return error

# finding rmse for predicted data
rmse1 = rmse(predictions,test)
print("RMSE value is :",rmse1)

# making the function which calculates MAPE value 
def mape(x_pred,x):
    sum = 0
    for i in range(len(x)):
        sum += (abs(x[i]-x_pred[i]))/x[i]
    error = sum*100/len(x)
    return error

# finding MAPE value for predicted data
mape_error = mape(predictions,test)
print("MAPE value is :",mape_error)


#### Question 3 ####
print("\n Question 3 :-")
lagged = [1,5,10,15,25] # array of lagged values

#initialising the array which can store rmse and mape values
rmse_values3 = []
mape_values3 = []

for k in lagged:
    pred = autoreg_model(train,test,k)
    rmse_values3.append(rmse(pred,test))
    mape_values3.append(mape(pred,test))

# converting 2D array into 1D array
rmse_values = [x[0] for x in rmse_values3]
mape_values = [x[0] for x in mape_values3]

print("RMSE Values are:")
print(rmse_values)
print("\n MAPE Values")
print(mape_values)
# plotting the bar graph for calculated RMSE values
plt.figure()
plt.bar(lagged,rmse_values)
plt.xlabel("Lag values")
plt.ylabel("RMSE values(in %)")
plt.show()

# plotting the bar graph for calculated MAPE values
plt.figure()
plt.bar(lagged,mape_values)
plt.xlabel("Lag values")
plt.ylabel("MAPE values")
plt.show()


#### Question 4 ####
print("\n Question 4 :-")
# As test and train arrays are 2D array. so we are converting test and train arrays into 1D  arrays (so that we can make it's pandas series)
train4 = [x[0] for x in train]
test4 = [x[0] for x in test]

# making pandas series of the test and train arrays so that we can give them as inputs to the autocorr function which calculates autocorrelation
train_df = pd.Series(train4)
test_df = pd.Series(test4)

# making the condition of 2/sqrt(T) 
t = 2/((len(train))**0.5)

# initialising the autocorrelation value with lag value as 1
a= autocorr(train_df,1)

p = 1  # p is the optimal lag value which is initialised to 1

# finding the optimal lag value
while(abs(a)>t):
    p += 1
    a = autocorr(train_df,p)
p = p - 1
print('Optimal number of lags: ',p)

# finding the predictions using the autoregression model defined above (with the help of optimal lag value)
optimal_predictions = autoreg_model(train,test,p)

# finding the rmse and MAPE values for the predictions with optimal lag value
rmse_optimal = rmse(optimal_predictions,test)
mape_optimal = mape(optimal_predictions,test)

print("RMSE value corresponding to optimal lag value: ",rmse_optimal)
print("MAPE value corresponding to optimal lag value:",mape_optimal)


#### Extra work ####
# now we will take complete data as the training data
# number of days for which value is to be predicted is the days between Oct3,2021 and Jan 31,2022 (both inclusive) that is 120 days

# defining autoregression model which take inputs as training data, number of days for which predictions are to be made(k) and the lag value(p)
def autoreg_model1(train,k,p):
    model = AR(train, lags=p) 
    # fit/train the model
    model_fit = model.fit() 
    # Get the coefficients of AR model
    coef = model_fit.params 
    
    history = train[len(train)-p:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(k):
        length = len(history)
        lag = [history[i] for i in range(length-p,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(p):
            yhat += coef[d+1] * lag[p-d-1] # Add other values
    
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(yhat) # append the value calculated now to history s it will be used to calculte the next value
    return predictions
# finding the predictions for dayes between Oct3,2021 and Jan 31,2022 (both inclusive)
predictions_new = autoreg_model1(X,120,p)

# plotting the line plot of the predicted values
plt.figure()
plt.plot(predictions_new)
matplotlib.dates.DateFormatter('%d')
plt.show()





#%%