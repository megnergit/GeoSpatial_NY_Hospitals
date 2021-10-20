# |------------------------------------------------------------------
# | # Geospatial Data Exercise
# |------------------------------------------------------------------
# |
# | This is an exercise notebook for the fifth lesson of the kaggle course
# | ["Geospatial Analysis"](https://www.kaggle.com/learn/geospatial-analysis)
# | offered by Alexis Cook and Jessica Li. The main goal of the lesson is
# | to get used to __Proximity Analysis__, using `geopandas` methods such as
# | `.distance`. We also learn how to use
# | `.unary_union` to connect multiple `POLYGON`s into one.

# | ## 1. Task
# |
# |   Every day someone injured in New York City in a car accident.
# |   If an ambulance can quickly rush into a nearby hospital with a patient
# |   is a matter of life and death. We will review the records of daily car
# |   crashes in New York City and the locations of hospitals there.
# |
# | 1. Find out if there is any vulnerable districts where
# |    it takes particularly long to transport the injured to a hospital.
# | 2. Create a recommender system to tell ambulance drivers
# |    to which hospital (the nearest) they should transport the injured.
# |

# | ## 2. Data
# |
# | 1. Daily records of car crashes in New York City.
# |
# | 2. Locations of hospitals in New York City.
# |
# | 3. General underlying  map.


# | ## 3. Notebook
# -------------------------------------------------------
# | Import packages.

import folium
import numpy as np
from folium import Marker, GeoJson
from folium.plugins import HeatMap
import pandas as pd
import geopandas as gpd
import plotly.graph_objs as go
from kaggle_geospatial.kgsp import *
from datetime import datetime
import os
from pathlib import Path

# -------------------------------------------------------
# | Set up some directories.

CWD = '/Users/meg/git6/ny_hospitals/'
DATA_DIR = '../input/geospatial-learn-course-data/'
KAGGLE_DIR = 'alexisbcook/geospatial-learn-course-data'
GEO_DIR = 'geospatial-learn-course-data'

os.chdir(CWD)

set_cwd(CWD)
set_data_dir(DATA_DIR, KAGGLE_DIR, GEO_DIR, CWD)
show_whole_dataframe(True)

# -------------------------------------------------------
# | Read the daily records of car crashes in New York City.

collisions_dir = DATA_DIR + \
    'NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions/'

collisions = gpd.read_file(
    collisions_dir + 'NYPD_Motor_Vehicle_Collisions.shp',
    parse_dates=['DATE', 'TIME'])

print(collisions.info())

# | It looks like, `parse_dates` does not convert `dtype` of
# | date and time to `datetime` from `object` (string).
# | Do it separately.

collisions['DATE'] = pd.to_datetime(collisions['DATE'])
collisions['TIME'] = pd.to_datetime(collisions['TIME'])

# -------------------------------------------------------
# | Let us start with the record in 2019 only.

print(collisions['DATE'].min())
print(collisions['DATE'].max())
print(len(collisions))

collisions = collisions[collisions['DATE'] >=
                        datetime.strptime('2019/01/01', '%Y/%m/%d')]

print(collisions['DATE'].min())
print(collisions['DATE'].max())
print(len(collisions))
collisions.head(3)

# -------------------------------------------------------
# | Read the locations of hospitals in New York City.

hospitals_dir = DATA_DIR + 'nyu_2451_34494/nyu_2451_34494/'
hospitals = gpd.read_file(hospitals_dir + 'nyu_2451_34494.shp')

print(hospitals.info())
print(hospitals.shape)
hospitals.head(3)

# -------------------------------------------------------
# Create a heatmap to show how the car crashes in New York City distributed.
# First, set up the center of the map, tiles, and the initial zoom-factor.

center = (collisions['LATITUDE'].mean(), collisions['LONGITUDE'].mean())
tiles = 'Stamen Terrain'
tiles = 'openstreetmap'
# tiles = 'cartodbpositron'
zoom = 12

m_1 = folium.Map(location=center, tiles=tiles, zoom_start=zoom)
HeatMap(data=collisions[['LATITUDE', 'LONGITUDE']],
        min_opacity=0.1,
        radius=15).add_to(m_1)
# --
embed_map(m_1, './html/m_1.html')
# --
show_on_browser(m_1, CWD + './html/m_1b.html')

# -------------------------------------------------------
# | There are concentrations of car crashes in
# | * Lower Manhattan
# | * Brooklyn
# | * The Bronx

# | in this order.

# -------------------------------------------------------
# | Let us overlay the locations of hospitals.

m_2 = folium.Map(location=center, tiles=tiles, zoom_start=zoom)
HeatMap(data=collisions[['LATITUDE', 'LONGITUDE']],
        min_opacity=0.1,
        radius=15).add_to(m_2)
dump = [Marker(location=[r['latitude'], r['longitude']], tooltip=r['name'],
               popup=r['address']).add_to(m_2) for i, r in hospitals.iterrows()]

# --
embed_map(m_2, './html/m_2.html')
# --
show_on_browser(m_2, CWD + './html/m_2b.html')

# -------------------------------------------------------
# | Pick up the cases that the closest hospitals are more than 10 km away.

# | 1. Add following columns to `collisions` table.
# | - name, id and address of the nearest hospital.
# | - distance to the nearest hospital.

# | 2. Flag it when the nearest hospitals is more than 10 km away.
# | Note that units of EPSG 2263 are meters.

print(hospitals.crs)
hospitals.crs == collisions.crs

# -------------------------------------------------------
id_nearest_h = []
name_nearest_h = []
address_nearest_h = []
distance_nearest_h = []

for i, c in collisions.iterrows():
    distance = hospitals['geometry'].distance(c['geometry']).min()
    idx = hospitals['geometry'].distance(c['geometry']).idxmin()

    id_nearest_h.append(hospitals.iloc[idx]['id'])
    name_nearest_h.append(hospitals.iloc[idx]['name'])
    address_nearest_h.append(hospitals.iloc[idx]['address'])
    distance_nearest_h.append(distance)

collisions['id_NEAREST_H'] = id_nearest_h
collisions['NAME_NEAREST_H'] = name_nearest_h
collisions['ADDRESS_NEAREST_H'] = address_nearest_h
collisions['DISTANCE_NEAREST_H'] = distance_nearest_h

print(collisions.info())
collisions.head(3)

# -------------------------------------------------------
# | How much is the fraction of car crashes
# | that the nearest hospitals are
# | more than 10 km away?

f_outside = (collisions['DISTANCE_NEAREST_H'] > 10 ** 4).mean()
print(
    f'\033[33mIn \033[96m{f_outside: 6.3f} \033[33m cases \
the nearest hospital is > 10 km away\033[0m')

# -------------------------------------------------------
# | Find out which part of New York City,
# | such cases often happen.
# | Use `unary_union` that we learned in the previous lesson.

ten_km_buffer = gpd.GeoDataFrame(geometry=hospitals.buffer(10 ** 4))
ten_km_buffer = ten_km_buffer.to_crs(epsg=4326)
ten_km_union = ten_km_buffer.unary_union

# We do not need to add crs to `ten_km_union` as it is a single object
# `MultiPolygon`, not  a `GeoDataFrame`.

# -------------------------------------------------------


def style_function(x):

    return {'fillColor': 'salmon',
            'stroke': True, 'color': 'salmon', 'weight': 8,
            'fillOpacity': 0.2}  # 'dashArray' :  '5,5'


tiles = 'openstreetmap'
tiles = 'Stamen Terrain'
m_3 = folium.Map(location=center, tiles=tiles, zoom_start=zoom)

HeatMap(data=collisions[['LATITUDE', 'LONGITUDE']],
        min_opacity=0.1,
        radius=15).add_to(m_3)

GeoJson(data=ten_km_union.__geo_interface__,
        style_function=style_function).add_to(m_3)

dump = [Marker([h['latitude'], h['longitude']],
               tooltip=h['name'],
               popup=h['address']).add_to(m_3)
        for i, h in hospitals.iterrows()]

folium.LatLngPopup().add_to(m_3)
# --
embed_map(m_3, './html/m_3.html')
# --
show_on_browser(m_3, CWD + './html/m_3b.html')

# -------------------------------------------------------
# | There are three sites that the new
# | hospitals should be build.
# | 1. Along NY 27 at the northeast of the JFK airport.
# | 2. Interchange of NY 24 at the south of Belmont Park.
# | 3. East edge of Brooklyn at NY 27A.

# -------------------------------------------------------
# | Let us add two hospitals from our hypothesis.
# | How much is the fraction of the car crashes now that happen
# | outside of 10 km buffer of the NYC hospitals?
# |
# | From the reading of pop-up on the map,

h_1 = [-73.7691, 40.6679]
h_2 = [-73.8443, 40.6714]

new_hospitals = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
    [h_1[0], h_2[0]], [h_1[1], h_2[1]], crs='epsg:4326'))

new_buffers = gpd.GeoDataFrame(
    geometry=new_hospitals.to_crs(epsg=2263).buffer(10 ** 4))

new_buffers = new_buffers.to_crs(epsg=4236)

# -------------------------------------------------------


def style_function2(x):

    return {'fillColor': 'maroon',
            'stroke': True, 'color': 'maroon', 'weight': 8,
            'fillOpacity': 0.2}  # 'dashArray' :  '5,5'


tiles = 'openstreetmap'
m_4 = folium.Map(location=center, tiles=tiles, zoom_start=zoom)

HeatMap(data=collisions[['LATITUDE', 'LONGITUDE']],
        min_opacity=0.1,
        radius=15).add_to(m_4)

GeoJson(data=ten_km_union.__geo_interface__,
        style_function=style_function).add_to(m_4)

GeoJson(data=new_hospitals.__geo_interface__).add_to(m_4)

GeoJson(data=new_buffers.__geo_interface__,
        style_function=style_function2).add_to(m_4)

dump = [Marker([h['latitude'], h['longitude']],
               tooltip=h['name'],
               popup=h['address']).add_to(m_4)
        for i, h in hospitals.iterrows()]

folium.LatLngPopup().add_to(m_4)
# --
embed_map(m_4, './html/m_4.html')
# --
show_on_browser(m_4, CWD + './html/m_4b.html')

# -------------------------------------------------------
# | Looks good.
# |

new_hospitals_ny = new_hospitals.to_crs(epsg=2263).copy()

all_hospitals = gpd.GeoDataFrame(pd.concat([hospitals['geometry'],
                                            new_hospitals_ny['geometry']], axis=0,
                                           ignore_index=True))

# It is enough to change the column name.

all_hospitals.rename(columns={0: 'geometry'}, inplace=True)
all_hospitals.crs = {'init': 'epsg:2263'}

distance_new = []
for i, c in collisions.iterrows():
    distance_new.append(
        all_hospitals['geometry'].distance(c['geometry']).min())

f2_outside = (np.array(distance_new) > 10 ** 4).mean()

print(
    f'\033[33mIn \033[96m{f2_outside: 6.3f} \033[33m cases \
the nearest hospital is > 10 km away\033[0m')

# -------------------------------------------------------
# | ## 4. Conclusion
# |
# | We tried both combinations above, i.e., (1,2) and (1,3).
# | We found (1,3) is more effective to reduce the fraction
# | of car crashes outside 10 km of a hospital.
# | Again the proposed sites are

h_1 = [-73.7691, 40.6679]
h_2 = [-73.8443, 40.6714]

# -------------------------------------------------------
# | ## 5. Appendix
# |
# | Write a function that takes longitude and latitude of a car crash,
# | and returns the name and the address of the nearest hospital.


def nearest_hospital(z, hospitals) -> gpd.GeoDataFrame:
    '''
    z : gpd.GeoDataFrame.  
        z['geometry'] contains the longitude and the latitude 
        of the place where the car crash happened. 

      hospitals : gpd.GeoDataFrame. A list of locations and other 
    informations (name, address, etc.) of the hospitals 
    in New York City.

    '''
    z = z.to_crs(epsg=2263)
    z = z.iloc[0]

    idx = hospitals['geometry'].distance(z['geometry']).idxmin()
    distance = hospitals['geometry'].distance(z['geometry']).min() / 10**3

    print(f'DISTANCE : \033[96m {distance:5.1f} \033[0mkm')

    return hospitals.iloc[idx][['name', 'address', 'longitude', 'latitude']]

# -------------------------------------------------------
# | Test.


# Imaginative crash location.
z = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
    [-73.9683], [40.7937], crs='epsg:4326'))

nearest_hospital(z, hospitals)

# -------------------------------------------------------
# | END
