# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 08/08/26
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```py
# Import necessary modules and functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Load dataset

data = pd.read_csv('/content/us-counties-recent.csv')


# Convert date column to datetime

data['date'] = pd.to_datetime(data['date'])


# Group total COVID cases by date

daily_data = data.groupby('date')['cases'].sum().reset_index()


# Sort data by date

daily_data = daily_data.sort_values('date')


# Declare required variables

N = 1000


# Set figure size

plt.rcParams['figure.figsize'] = [12, 6]


# Extract cases data

X = daily_data['cases']


# Visualize the original data

plt.plot(X)
plt.title('Original Data (COVID-19 Cases)')
plt.show()


# Plot ACF and PACF of original data

plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)//4, ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()


# Fit ARMA(1,1) model

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()


# Extract ARMA(1,1) parameters

phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']


# Simulate ARMA(1,1) process

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)


# Plot simulated ARMA(1,1)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()


# Plot ACF and PACF for ARMA(1,1)

plot_acf(ARMA_1)
plt.show()

plot_pacf(ARMA_1)
plt.show()


# Fit ARMA(2,2) model

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()


# Extract ARMA(2,2) parameters

phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']


# Simulate ARMA(2,2) process

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)


# Plot simulated ARMA(2,2)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()


# Plot ACF and PACF for ARMA(2,2)

plot_acf(ARMA_2)
plt.show()

plot_pacf(ARMA_2)
plt.show()
```

## OUTPUT:
SIMULATED ARMA(1,1) PROCESS:
<img width="986" height="517" alt="image" src="https://github.com/user-attachments/assets/13a7614c-6c75-499c-923d-ce491d0cd90b" />



Partial Autocorrelation
<img width="992" height="517" alt="image" src="https://github.com/user-attachments/assets/ccfba97c-233d-4cec-98cb-05e3e7cb12e2" />

Autocorrelation
<img width="993" height="538" alt="image" src="https://github.com/user-attachments/assets/501c3963-4d91-45f3-873c-f26ff4bd0c15" />



SIMULATED ARMA(2,2) PROCESS:
<img width="990" height="517" alt="image" src="https://github.com/user-attachments/assets/61b8fd9e-c878-4936-be8b-8d594202c630" />

Partial Autocorrelation
<img width="987" height="518" alt="image" src="https://github.com/user-attachments/assets/c8143557-71af-45bd-8f27-6ad4d50fa6bc" />

Autocorrelation
<img width="992" height="517" alt="image" src="https://github.com/user-attachments/assets/f7e53b0d-de4e-4be8-9b81-947eaca54709" />

## RESULT:
Thus, a python program is created to fir ARMA Model successfully.
