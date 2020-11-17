import streamlit as st
import numpy as np
import pandas as pd
import datetime
import altair as alt
from vega_datasets import data
import copy

@st.cache(persist=True)

def load_data(path):
    data=pd.read_csv(path)
    return data

#load in data
fb_mask_original=load_data("fb_mask.csv")
fb_sympton_original=load_data("fb_sympton.csv")
fb_sympton=copy.deepcopy(fb_sympton_original)
fb_mask=copy.deepcopy(fb_mask_original)
fb_mask['time_value']= pd.to_datetime(fb_mask['time_value'], format='%Y/%m/%d')
fb_sympton['time_value']= pd.to_datetime(fb_sympton['time_value'], format='%Y/%m/%d')
fb_mask.rename(columns={'value':'mask_percentage'}, inplace=True)
fb_sympton.rename(columns={'value':'sympton_percentage'}, inplace=True)

fb_all=fb_mask.merge(fb_sympton, on=['time_value','geo_value'])
fb_all=fb_all[['geo_value','time_value','mask_percentage','sympton_percentage']]
fb_all = fb_all[fb_all['time_value']>'2020-09-08']

states=fb_all.geo_value.str.upper().unique()

#first plot: correlation between wearing mask and having symptons
st.title("Let`s see the correlation between wearing mask and having symptons.")

state_choice = st.sidebar.multiselect(
    "Which state are you interested in?",
    states.tolist(), default=['AK','AL','AR','AZ','CA','CO']
)

date_range = st.sidebar.date_input("Which range of date are you interested in? Choose between %s and %s"% (min(fb_all['time_value']).strftime('%Y/%m/%d'),  max(fb_all['time_value']).strftime('%Y/%m/%d')), [min(fb_all['time_value']), max(fb_all['time_value'])])

fb_temp = fb_all[fb_all['geo_value'].str.upper().isin(state_choice)]

if len(date_range)==2:
    fb_selected = fb_temp[(fb_temp['time_value']>=pd.to_datetime(date_range[0])) & (fb_temp['time_value']<=pd.to_datetime(date_range[1]))]
else:
    fb_selected = fb_temp[(fb_temp['time_value']>=pd.to_datetime(date_range[0]))]

scatter_chart = alt.Chart(fb_selected).mark_circle().encode(
    x=alt.X('mask_percentage', scale=alt.Scale(zero=False), axis=alt.Axis(title='percentage of wearing masks')), 
    y=alt.Y('sympton_percentage', scale=alt.Scale(zero=False), axis=alt.Axis(title='percentage of having covid symptons'))
)
scatter_chart + scatter_chart.transform_regression('mask_percentage', 'sympton_percentage').mark_line()



map_data = fb_all[fb_all['time_value']==pd.to_datetime(date_range[0])].copy()
ids = [2,1,5,4,6,8,9,11,10,12,13,15,19,16,17,18,20,21,22,25,24,23,26,27,29,28,30,37,38,31,33,34,35,32,
       36,39,40,41,42,44,45,46,47,48,49,51,50,53,55,54,56] 
map_data['id'] = ids

states = alt.topo_feature(data.us_10m.url, 'states')
variable_list = ['mask_percentage','sympton_percentage']

chart = alt.Chart(states).mark_geoshape().encode(
    alt.Color(alt.repeat('row'), type='quantitative')
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(map_data, 'id', variable_list)
).properties(
    width=500,
    height=300
).project(
    type='albersUsa'
).repeat(
    row=variable_list
).resolve_scale(
    color='independent'
)

st.write(chart)
