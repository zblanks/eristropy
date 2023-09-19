## Overview and Introduction
Welcome to the EristroPy package documentation! This document provides an overview of EristroPy, 
a powerful framework for working time series signals via entropy using Python. 

The EristroPy package is designed to streamline the variability analysis of signals in Python.
It provides end-to-end functionality, starting from constructing stationary signals 
to determining appropriate metric parameters and efficiently computing entropy 
and variability measures. By leveraging EristroPy, researchers and practitioners can 
focus on the analysis and interpretation of time series data, rather than spending time 
and effort on the intricate details of signal processing and analysis.

To the best of our knowledge, EristroPy is the only existing solution in Python that 
offers all the necessary functionality for valid and reproducible entropy analysis 
using novel and scalable heuristics. Its features and benefits enable researchers 
to perform comprehensive variability analysis, gain valuable insights, and 
contribute to advancements in the field of cardiopulmonary exercise testing.

## Features & Benefits
EristroPy offers a range of features and benefits that facilitate the analysis of time series signals:

* **Automatic Signal Stationarity**: EristroPy enables seamless construction of stationary signals, 
a necessary condition for valid entropy and variability analysis. 
It incorporates two common techniques, differencing and de-trending, 
and performs statistical stationarity checks to ensure that the dataset contains valid signals.
* **Scalable Entropy Calculations**: EristroPy provides efficient implementations of 
sample and permutation entropy. Leveraging Numba's just-in-time compilation scheme, 
EristroPy ensures fast and scalable computations, allowing researchers to focus on 
the analysis rather than the intricacies of the calculations.
* **Optimal Parameter Selction**: Determining appropriate parameter settings for 
entropy measures can be challenging. EristroPy takes the guesswork out by providing 
reasonable recommendations based on rigorous, nonparametric statistical approaches. 
These recommendations empower researchers to confidently choose suitable parameters for their analysis.

## Installation
You can install EristroPy by using pip:

```bash
pip install eristropy
```

## Usage
To start using EristroPy in your Python project, import it as follows:

```python
import eristropy
```

## License
EristroPy is released under the MIT License.

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
