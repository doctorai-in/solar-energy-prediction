# Prediction of solar energy using Machine Learning 
 
# Introduction: 
* `Renewable energies have been gaining traction in the last decades. Non-polluting, and not limited by resources, these energies would be an ideal source of electricity for any activity, whether domestic or industrial, if it was not because of their unreliability. Indeed, renewable energies throughput varies significantly depending on the conditions and the characteristics of the place where they are located, which makes it difficult to estimate how much benefit in terms of power is going to be obtained from them.`
* `Photovoltaic systems have become an important source of renewable energy generation. Because solar power generation is intrinsically highly dependent on weather fluctuations, predicting power generation using weather information has several economic benefits, including reliable operation planning and proactive power trading. In this project Machine learning models need to be created that predicts the amounts of solar power generation using weather information. From this exercise, we should aim to understand how different factors affect the solar power generation.` 

# Dataset:
 * `The time averaged values of each feature have been done to obtain a consistent hourly resolution. In order to obtain a robust estimate of the weather features of the solar farm region, we spatially averaged them over the 3 closest weather stations using the barycenter formula.`
 
| Weather feature | Unit |
| --- | --- | 
|Cloud coverage | % Range | 
|Visibility | Miles |
|Temperature | Celcius | 
|Dew point | Celcius | 
|Relative humidity | % | 
|Wind speed | Mph | 
|Station pressure  | inchHg | 
|Altimeter | inchHg | 
|Solar energy | KWh | 

* `The time-series for both solar energy and weather parameters starts on February 1, 2016 and ends on October31, 2017.We randomized our dataset and divided the data into 80%-10%-10% partitions for the training, development, and test sets.`
 
# Conclusion:
The one with the least root mean squared value shall be regarded as the best model. 
