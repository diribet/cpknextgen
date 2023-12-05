# C<sub>pk</sub> NextGen
[![Build](https://github.com/diribet/cpknextgen/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/diribet/cpknextgen/actions/workflows/test.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/cpknextgen)](https://pypi.org/project/cpknextgen/)

## Process Capability using Normal distribution mixture
This procedure allows you to compute the $c_p$ and $c_{pk}$ process capability indices, utilizing an innovative method for estimating the center and quantiles of the process, including their uncertainty. The advantage of this approach is its consistency in comparing processes and their changes over time, without relying heavily on anecdotal evidence or having to categorize the “process type”.

## Usage
Install the library
```shell
python -m pip install cpknextgen
```

Import the `evaluate` function which is the main entry point

```python
from cpknextgen import evaluate
``` 
The function takes in 1-dimensional list/numpy array of process data, the tolerances (as mentioned above) and some other parameters for the Gaussian Mixture. Refer to the docstring for further details. It calculates point and interval estimates of the indices, and graphical results - empirical CDF and sample estimate of CDF.

## Methodology
The method leverages past performance data and experience. The process of calculation, as outlined in the figure below, can either include or exclude the prior, where the prior is a dataset from past performance used as Bayesian-type information.

![method_illustration](https://github.com/diribet/cpknextgen/assets/1448805/28878d30-307a-40b5-8b04-388eb6f05d96)

The algorithm is designed for continuous process observation, meaning it estimates the resulting indices' value with uncertainty at each point of the process. It can predict what the resulting indices' values will be when the process is complete for a given production period (e.g., a shift, a day, a week, etc.).

### Calculation Without Prior
Calculation without the prior is equivalent to estimating the indices on the prior, and the resulting information can be used to calculate the indices on another dataset with this prior. It is especially recommended for “closed” production periods, such as calculating the process capability for a recently concluded shift.

The data is often accompanied by varying amounts of contextual information, most notably the tolerance limits and the extreme limits. These extreme limits are dictated by physical restrictions or plausibility limits and are not mandatory. Any data outside these limits are treated as outliers and ignored.
To calculate $c_{pk}$, at least one tolerance limit is necessary. Both tolerance limits are needed for a proper calculation of c_p. If not provided, the algorithm only estimates the quantiles, giving the process center and width, without a tolerance interval for comparison.

Before distribution estimation, data transformation based on shape takes place. This involves the following steps:
1. Logarithmic or logit transformations based on extreme limits, when they exist.
2. Applying a Yeo-Johnson transformation.
3. Scaling the tolerance interval to a +/-1 interval. In cases where one or both tolerances are missing, they are estimated as "tolerance quantiles" from the data.

### Calculation With Prior (NOT IMPLEMENTED!)
The data transformation method is derived from the prior. The extent to which the prior is used in distribution estimation varies, depending on the amount of information available at the time of estimation. With limited information, e.g., after the first hour of an 8-hour shift, there is a higher reliance on the past shape of the process from the prior. As the shift progresses, indices will be estimated purely from the information from the ongoing production period.

This balance is controlled by the "Basic sample size" and the "Process Length" parameters. Regardless of the size of the prior, the algorithm ensures the amount of information derived from it corresponds to these two parameters. Hence, it is advisable to use a "sufficiently large" prior dataset that includes all reasonable process variants.

### Special Cases
There are two types of special cases that limit the calculation. In the first scenario, no calculation proceeds if there's only one data point or if all data points in the set have the same value. In the second scenario, the calculation proceeds, but it does not produce a prior that can be used for another dataset, e.g., when the lower limit/tolerance isn't given, and all data are above the upper tolerance. 
These special cases are currently under review, and we look forward to sharing updated methodologies to handle them in the future.

### Conclusion
![prior_illustration](https://github.com/diribet/cpknextgen/assets/1448805/c56a1653-d0bf-42ce-8970-bb714be48e98)

This novel method for computing process capability indices offers a more consistent and data-driven approach. Feedback and contributions are encouraged as we continue to refine and extend this methodology. Please refer to the figure above for a graphical representation of the process.
