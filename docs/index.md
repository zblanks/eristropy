# Overview and Introduction
Welcome to the CPyET Package Documentation! This document provides an overview of CPyET, 
a powerful framework for working with cardiopulmonary exercise testing (CPET) signals via Python. 
CPET is a widely used method for assessing cardiovascular and pulmonary function during exercise. 
It involves monitoring various physiological signals, such as heart rate, oxygen consumption, 
and breathing patterns, to evaluate an individual's fitness level and diagnose 
certain cardiovascular and respiratory conditions.

Analyzing CPET signals can provide valuable insights into the dynamic changes 
that occur within the cardiopulmonary system during exercise. 
It helps researchers, clinicians, and fitness professionals understand the 
efficiency, limitations, and adaptive responses of the cardiovascular and respiratory systems. 
By examining the patterns and variability of these signals, researchers can gain
a deeper understanding of physiological mechanisms, identify abnormalities or impairments, 
and monitor the effectiveness of interventions or training programs.

The CPyET package is designed to streamline the variability analysis of CPET signals in Python.
It provides end-to-end functionality, starting from constructing stationary signals 
to determining appropriate metric parameters and efficiently computing entropy 
and variability measures. By leveraging CPyET, researchers and practitioners can 
focus on the analysis and interpretation of CPET data, rather than spending time 
and effort on the intricate details of signal processing and analysis.

To the best of our knowledge, CPyET is the only existing solution in Python that 
offers all the necessary functionality for valid and reproducible CPET analysis 
using novel and scalable heuristics. Its features and benefits enable researchers 
to perform comprehensive variability analysis, gain valuable insights, and 
contribute to advancements in the field of cardiopulmonary exercise testing.

# Features & Benefits
CPyET offers a range of features and benefits that facilitate the analysis of CPET signals:

* **Automatic Signal Stationarity**: CPyET enables seamless construction of stationary signals, 
a necessary condition for valid entropy and variability analysis[^1]. 
It incorporates two common techniques &mdash differencing and de-trending &mdash 
and performs statistical stationarity checks to ensure that the dataset contains valid signals.
* **Scalable Entropy Calculations**: CPyET provides efficient implementations of 
sample and permutation entropy. Leveraging a Numba's just-in-time compilation scheme, 
CPyET ensures fast and scalable computations, allowing researchers to focus on 
the analysis rather than the intricacies of the calculations.
* **Optimal Parameter Selction**: Determining appropriate parameter settings for 
entropy measures can be challenging. CPyET takes the guesswork out by providing 
reasonable recommendations based on rigorous, nonparametric statistical approaches. 
These recommendations empower researchers to confidently choose suitable parameters for their analysis.

# Installation
CPyET is not yet available on PyPI. However, once it is published, you will be able to install it using pip:

```python
pip install cpyet
```

# Usage
To start using CPyET in your Python project, import it as follows:

```python
import cpyet
```

# License
CPyET is released under the MIT License.

The MIT License (MIT)

Copyright (c) 2023 Zachary Blanks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


[^1]: Chatain, Cyril, et al. "Effects of nonstationarity on muscle force signals regularity during a fatiguing motor task." 
IEEE Transactions on Neural Systems and Rehabilitation Engineering 28.1 (2019): 228-237.