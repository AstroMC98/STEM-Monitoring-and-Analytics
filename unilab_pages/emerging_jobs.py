# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 00:04:17 2022
"""
# --------------- IMPORTS ------------------
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import altair as alt

import pandas as pd
import numpy as np

# ---------------- SETUP -------------------
data_folder = "data/"
img_folder = "images/"
mid_width = 5.1

# ---------------FILES ---------------------
df_cagr = pd.read_csv(f"{data_folder}/4_all_occupation_stem.csv")
df_cagr = df_cagr.rename(columns={'Unnamed: 0':'rank', 'occupation':'Occupation'})
df_cagr['rank'] = df_cagr['rank']+1
df_cagr['CAGR'] = (df_cagr['CAGR']*100).round(decimals=2)

#st.dataframe(df_cagr.sort_values('CAGR', ascending=False).head(15))

df_count = pd.read_csv(f"{data_folder}/4_occupation_counts.csv")
df_count = df_count.rename(columns={'Onet Occupation': 'Occupation'})
df_count = pd.merge(df_count, df_cagr[['rank', 'Occupation']], on='Occupation').sort_values('rank')
df_count = df_count.iloc[:, 1:].copy()

#st.dataframe(df_count.head(15))


# -------------- HELPER FUNCTIONS ----------
def plot_line_temp():
    chart_data = pd.DataFrame(
         np.random.randn(20, 3),
         columns=['a', 'b', 'c'])

    st.line_chart(chart_data)

def plot_temp():

    # Add histogram data
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2

    # Group data together
    hist_data = [x1, x2, x3]

    group_labels = ['Group 1', 'Group 2', 'Group 3']

    # Create distplot with custom bin_size
    fig = ff.create_distplot(
             hist_data, group_labels, bin_size=[.1, .25, .5])

    # Plot!
    st.plotly_chart(fig, use_container_width=True)


def plot_hbar_topN(cagr_or_count, n=10):
    """Plot the Top N Jobs on CAGR or Total counts"""

    if cagr_or_count == 'Growth (CAGR)':
        df_ = df_cagr.head(n).sort_values('rank', ascending=False)
        fig = px.bar(df_, x='CAGR', y='Occupation',
                     orientation='h',
                     title=f"Compounded Annual Growth (CAGR) for Top {n} Occupations",
                     labels={'CAGR':'CAGR (%)'}
                     )

    elif cagr_or_count == 'Job Count (Total 2017-2021)':
        st.write("NOTE: Consider revising to stacked bar chart for each year")
        df_ = df_count.head(n).sort_values('rank', ascending=False)
        fig = px.bar(df_, x='Total', y='Occupation',
                     orientation='h',
                     title=f"Total Job Postings for Top {n} Occupations",
                     )
    else:
        st.write("ERROR: No such option")

    fig.update_traces(marker_color='#30BBB1')
    st.plotly_chart(fig, use_container_width=True)

def plot_hbar_topN_range(n_range=(1,10)):
    """Plot the Top N Jobs on CAGR"""

    # st.dataframe(df_cagr[(df_cagr.rank >= n_range[0]) & (df_cagr.rank <= n_range[1])])
    df_ = df_cagr.iloc[n_range[0]-1:n_range[1]-1].sort_values('rank', ascending=False)
    fig = px.bar(df_, x='CAGR', y='Occupation',
                 orientation='h',
                 title="Compounded Annual Growth (CAGR) for Selected Occupations",
                 labels={'CAGR':'CAGR (%)'}
                 )

    fig.update_traces(marker_color='#30BBB1')
    st.plotly_chart(fig, use_container_width=True)


def plot_scatter_growth_count(df, year):

    c = (alt.Chart(df).mark_circle()
         .encode(
             #x='Count (log)',
             x=alt.X('Count (log)', scale=alt.Scale(domain=(-1, 8))),
             y=alt.Y('CAGR', scale=alt.Scale(domain=(-60, 60))),
             #size='size',
             #color='top10',
             color=alt.Color('top10', scale=alt.Scale(domain=[True, False], range=['#FDC13A', 'Grey'])),
             tooltip=['Occupation', 'CAGR', year])
         .properties(width=500,height=500,
                     title='Job Growth from 2017 to 2021 against 2021 Job Counts',)
         .configure_mark(opacity=1, size=300)
         )

    st.altair_chart(c, use_container_width=True)



def plot_line_trend(top_n, occ_selected, top10_or_select):
    df_trend = df_count[['Occupation', '2017', '2018', '2019', '2020', '2021']]
    df_trend = pd.melt(df_trend, id_vars=['Occupation'], value_vars=['2017', '2018', '2019', '2020', '2021'])
    df_trend.columns = ['Occupation', 'Year', 'Count']

    fig_trend = go.Figure(layout=go.Layout(title=f"Job Count Trend for {top10_or_select}"))
    for j in occ_selected: #df_trend.Occupation.unique():
        this_trend = df_trend[df_trend.Occupation == j]
        fig_trend.add_trace(go.Scatter(name=j,
                                       x=this_trend['Year'],
                                       y=np.log(this_trend['Count']))
                            )
    fig_trend.update_layout(template='simple_white',
                            autosize=True,
                            xaxis_title="Year of Job Posting",
                            yaxis_title="Count of Jobs (log)",
                            width=500, height=700)
    st.plotly_chart(fig_trend, use_container_width=True)

# -------------- LAYOUT --------------------
#############
## SIDEBAR ##
#############
#st.sidebar.image(f"{img_folder}/SLA_PH_Logo/Main-Square.png", use_column_width=True)

##############
##  HEADER  ##
##############
def get_contents():
    st.header("Branching from STEM Education")
    st.write("Correlating the Curricula and STEM Skills Necessary to Support In-Demand Jobs in the Philippines")

    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
    with row0_1:
        st.title('Emerging Jobs in the Philippines 2021')


    with row0_2:
        #st.image(f"{img_folder}/SLA_PH_Logo/Main-Horizontal.png", use_column_width=True)
        st.image(f"{img_folder}/SLA_PH_Logo/Black/Black-Horizontal.png", use_column_width=True)
        st.write('')

    row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, mid_width, .1))
    with row1_1:
        h_text = open(f"{data_folder}/4_header.txt","r+")
        h_text = "".join(h_text.readlines())
        st.markdown(h_text)

    ###############
    ##  CONTENT  ##
    ###############

    #--- FIGURE: Trend ---
    row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, mid_width, .1))
    with row2_1:
        st.header("Online Job Vacancy (OJV) Count")
        fig1_text = open(f"{data_folder}/4_fig1_trends.txt","r+")
        fig1_text = "".join(fig1_text.readlines())
        st.markdown(fig1_text)

    row2a_spacer1, row2a_1, row2a_spacer2, row2a_2, row2a_spacer3  = st.columns((.2, 2.3, .1, 5.4, .2))

    with row2a_1:
        top10_or_select = st.radio(
            "Which occupations do you want to see?",
            ('Top N Emerging Occupations', 'Own Selection'))

    with row2a_2:
        if top10_or_select == 'Own Selection':
            occ_selected = st.multiselect(
                 'Select occupations to include:',
                 df_count['Occupation'].tolist(),
                 [])
            #occ_selected = df_count.head(5)['Occupation'].tolist()
            st.write(f"{len(occ_selected)} of {len(df_count)} Selected")
            top_n = 200
        elif top10_or_select == 'Top N Emerging Occupations':
            top_n = st.slider('Select up to Top 30 Occupations', 5, 30, 10)
            occ_selected = df_count.head(top_n)['Occupation'].tolist()
        else:
            st.write("Invalid Selection")


    row2b_spacer1, row2b_1, row2b_spacer2 = st.columns((.1, mid_width, .1))

    with row2b_1:
        plot_line_trend(top_n, occ_selected, top10_or_select)


    #--- FIGURE: Top 10 Horizontal Bar ---
    row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, mid_width, .1))
    with row3_1:
        st.header("Emerging Jobs by Growth")
        fig2_text = open(f"{data_folder}/4_fig2_hbar.txt","r+")
        fig2_text = "".join(fig2_text.readlines())
        st.markdown(fig2_text)

    row3a_spacer1, row3a_1, row3a_spacer2, row3a_2, row3a_spacer3  = st.columns((.2, 2.3, .1, 5.4, .2))
    with row3a_1:
        st.write('')
        st.write('')
        st.write('Select Range:')
        min_n_range = st.number_input('Minimum Rank', min_value = 1, max_value = len(df_cagr)+1, value = 1, step = 1)
        max_n_range = st.number_input('Maximum Rank', min_value = 1, max_value = len(df_cagr)+1, value = 10, step = 1)
        top_n_range = (min_n_range,max_n_range)
        #top_n_range = st.select_slider('Select Record by Emerging Rank', range(1, len(df_cagr)+1),(1, 10))

    with row3a_2:
        plot_hbar_topN_range(top_n_range)

    #--- FIGURE: Scatter plot ---
    row4_spacer1, row4_1, row4_spacer2 = st.columns((.1, mid_width, .1))
    with row4_1:
        st.header("Comparison of Growth and Demand")
        fig3_text = open(f"{data_folder}/4_fig3_scatter.txt","r+")
        fig3_text = "".join(fig3_text.readlines())
        st.markdown(fig3_text)

    row4a_spacer1, row4a_1, row4a_spacer2, row4a_2, row4a_spacer3  = st.columns((.2, 4, .1, 4, .2))
    with row4a_1:
        year = '2021'
        df = pd.merge(df_cagr[['Occupation', 'CAGR', 'rank']], df_count[['Occupation', year, '2017']], on='Occupation', how='inner')
        df['top10'] = df['rank'] < 11
        df['Count (log)'] = np.log(df[year])
        df['size']=20
        df = df.set_index('rank')

        st.dataframe(df[['Occupation', '2017', '2021', 'CAGR']].rename(columns={'2017': '2017 (Count)', 
                                                                                '2021': '2021 (Count)',
                                                                                'CAGR': 'CAGR (%)'
                                                                                }),
                     width=700)

    with row4a_2:
        plot_scatter_growth_count(df, year)
