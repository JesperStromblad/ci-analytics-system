# Analytics system for Continuous Integration of Python Projects

This repository contains front-end code for analyzing Python-based projects. The front-end is developed as a part of the research work conducted for improving performance testing during DevOps practices at CERN.

# Instructions for setting up the analytic system.

## Back-end deployment
There are different steps which needs to be taken in order to deploy back-end. 
- **CI Data Streaming** The back-end of the analytic system takes analysis data from CI process via PerfCI plugin. More information about PerfCI and how it can be setup with your Python-project can be found in the [article] (https://ieeexplore.ieee.org/document/9286019) and the code is available on [GitHub] (https://github.com/JesperStromblad/perfci).
 > Example code for profiling program resource e.g., execution time and memory utilization can be found in the [code] (https://github.com/JesperStromblad/perfci/blob/main/plugins/resourcecollector.py)
> More complex profiling such as measuring code metrics can be found in [tracer plugin] (https://github.com/JesperStromblad/perfci/blob/main/plugins/tracer.py)

