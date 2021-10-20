# Geospatial Data Exercise G6

This is an exercise notebook for the fifth lesson of the kaggle course
["Geospatial Analysis"](https://www.kaggle.com/learn/geospatial-analysis)
offered by Alexis Cook and Jessica Li. The main goal of the lesson is
to get used to __Proximity Analysis__, using `geopandas` methods such as
`.distance`. We also learn how to use 
`.unary_union` to connect multiple `POLYGON`s into one.

------------------------------------------------------------------
## How to run the Demo

1. Load `exercise-proximity-analysis.ipynb` to Jupyter Notebook and run, or

2. `> python3 exercise-proximity-analysis.py'

------------------------------------------------------------------
## Task


   Every day someone injured in New York City in a car accident.
   If an ambulance can quickly rush into a nearby hospital with a patient
   is a matter of life and death. We will review the records of daily car
   crashes in New York City and the locations of hospitals there.

 1. Find out if there is any particularly vulnerable districts where
    it takes longer to transport the injured to a hospital.

 2. Create a recommender system to tell ambulance drivers
    to which hospital (the nearest) they should transport the injured.


------------------------------------------------------------------
## Directory Tree
```
.
├── LICENSE
├── README.md
├── README.md~
├── exercise-proximity-analysis.ipynb
├── exercise-proximity-analysis.py
├── html
│   ├── m_1.html
│   ├── m_1b.html
│   ├── m_2.html
│   ├── m_2b.html
│   ├── m_3.html
│   ├── m_3b.html
│   ├── m_4.html
│   └── m_4b.html
├── kaggle_geospatial
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── kgsp.cpython-38.pyc
│   └── kgsp.py
└── requirements.txt

```
* `html` directory in the repo is intentionally kept empty. It will be
   filled when the Python demo ran successfully. 
* kgsp is a python module that contains functions used in the exercise. 
------------------------------------------------------------------
END

