import pandas as pd
import geopandas as gpd

from pathlib import Path
import os
import webbrowser
import zipfile

import plotly.graph_objs as go
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster, HeatMap

# -------------------------------------------------------
# | Some housekeeping functions.
# | Later they will go to a module ``../kaggle_geospatial`.


def set_cwd(CWD):
    if Path.cwd() != CWD:
        os.chdir(CWD)

# | If we have not downloaded the course data, get it from Alexis Cook's
# | kaggle public dataset.


def set_data_dir(DATA_DIR, KAGGLE_DIR, GEO_DIR, CWD):
    if not Path(DATA_DIR).exists():
        command = 'kaggle d download ' + KAGGLE_DIR
        os.system(command)
        os.chdir(DATA_DIR)
#        with zipfile.ZipFile('geospatial-learn-course-data', 'r') as zip_ref:
        with zipfile.ZipFile(GEO_DIR, 'r') as zip_ref:
            zip_ref.extractall('.')
        os.chdir(CWD)

# | Some housekeeping stuff. Change `pandas`' options so that we can see
# | whole DataFrame without skipping the lines in the middle.


def show_whole_dataframe(show):
    if show:
        pd.options.display.max_rows = 999
        pd.options.display.max_columns = 99

# | This is to store the folium visualization to an html file, and show it
# | on the local browser.


def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')


def embed_plot(fig, file_name):
    from IPython.display import IFrame
    fig.write_html(file_name)
    return IFrame(file_name, width='100%', height='500px')


def show_on_browser(m, file_name):
    '''
    m   : folium Map object
    Do not miss the trailing '/'
    '''
    m.save(file_name)
    url = 'file://'+file_name
    webbrowser.open(url)

# -------------------------------------------------------
