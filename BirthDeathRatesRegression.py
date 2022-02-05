import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Source: https://stackoverflow.com/questions/60160803/scipy-optimize-curve-fit-for-logistic-function
def logistic_function(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b

# Source: https://calculator.academy/population-growth-calculator/
def population_growth(population, growth_rate, time):
    return (population * (pow(1 + growth_rate, time)))

##Reading in the data and cleaning it
with open('BirthsandDeathRates.csv', 'r') as f:
     lines = f.readlines()
birth_rates =[]
death_rates = []
for i in range(51, 73):
    #print(lines[i])
    birth_rates.append(float(lines[i].split(',')[2]))
    death_rates.append(float(lines[i].split(',')[3]))
years = list(range(2000, 2022))

for i in range(len(birth_rates)):
    birth_rates[i] = birth_rates[i]/1000000

for i in range(len(death_rates)):
    death_rates[i] = death_rates[i]/1000000


##Converting the arrays to numpy arrays and reshaping the x-axis as well as flattening the arrays into 1D as opposed to 2D
years = np.array(years).reshape(-1, 1).flatten()
birth_rates = np.array(birth_rates).flatten()
death_rates = np.array(death_rates).flatten()

paramsBirth, covarianceBirth = curve_fit(logistic_function, years, birth_rates, p0=[120, 2000, 3, 2010], maxfev=10000) # p0 is the initial guess for the parameters
r2 = r2_score(birth_rates, logistic_function(years, *paramsBirth)) # Calculating the R^2 value
print('R-Squared for births:', r2)

predictionBirth = logistic_function(2122, *paramsBirth)
print('Prediction for births in 2122:', round(predictionBirth*1000000), "people")

paramsDeath, covarianceDeath = curve_fit(logistic_function, years, death_rates, p0=[50, 2000, 5, 2010], maxfev=10000) # p0 is the initial guess for the parameters
r2 = r2_score(death_rates, logistic_function(years, *paramsDeath)) # Calculating the R^2 value
print('R-Squared for births:', r2)

predictionDeath = logistic_function(2122, *paramsDeath)
print('Prediction for deaths in 2122:', round(predictionDeath*1000000), "people")

# Final prediction for world population using birth
# and death rates calculated using a logistic function
forecasted_years = list(range(2021, 2122))
forecasted_births = []
forecasted_deaths = []
for i in range(len(forecasted_years)):
    if i == 2122:
        break
    forecasted_births.append(logistic_function(forecasted_years[i], *paramsBirth))
    forecasted_deaths.append(logistic_function(forecasted_years[i], *paramsDeath))

world_population = []
world_population.append(7923230649) # current world population in 2021
for i in range(len(forecasted_years)):
    growthRate = (forecasted_births[i] - forecasted_deaths[i]) / world_population[i]
    world_population.append(population_growth(world_population[i], growthRate, 1))


print('Prediction for world population in 2122:', round(world_population[-1]), "people")
