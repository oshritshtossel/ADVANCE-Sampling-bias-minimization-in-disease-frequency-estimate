# Sampling bias minimization in disease frequency estimate
The project provides two alternative unbiased estimation models to the daily probability of being positive to a pandemic, based on the results of tests between a specific period of time
and the profile of the individual.
## Datasets:
The analysis was done on data of the Israeli Covid-19 between the dates of 11.3.2020 and 18.10.2020 , provided by the Ministry of health in Israel.
The data can be found in the following link: https://data.gov.il/dataset/covid-19/resource/d337959a-020a-4ed3-84f7-fca182292308

We checked our models estimation by checking the correlation to the daily number of deaths. The information about the daily number of deaths in Israel was taken from the dashboard
of the Israeli Ministry of health. The data was collected manually from the following link: https://datadashboard.health.gov.il/COVID-19/general

## Our Models:
### Global Model
This model gives us an estimation to the daily probability to be positive to a pandemic based on bayesian inference, regarding the whole data in one step.
One can find the full details and explanations in our paper.

### Online Model
This model gives us an estimation to the daily probability to be positive to a pandemic based on bayesian inference, calculating the estimation of the first 28 days
in one step using the global model, but the estimations of the rest days are online.They are calculated day by day according to its previous estimations. 
One can find the full details and explanations in our paper.

## Usage:

