# Changepoint/Concept Drift Detection Ensemble
Changepoints are abrupt variations in the generative parameters of a data sequence. Online detection of changepoints is useful in modelling and prediction of time series in application areas such as finance, biometrics, and robotics. The model is designed to classify temporal sequences in an "online" manner, as to whether a changepoint occured. 
The model can be configured to identify changepoints as lower temporal granularities to serve changepoint detection purposes such as in financial or medical applications. The ensemble can also be configured at a higher temporal granularity to serve in a monitoring role as a sort of meta model that can be used to trigger retraining or event driven messaging systems.