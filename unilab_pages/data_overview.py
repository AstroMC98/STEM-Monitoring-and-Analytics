# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 20:38:09 2022

@author: Sarah Isabel Mendoza
"""

# --------------- IMPORTS ------------------
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
from pywaffle import Waffle
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

# ---------------- SETUP -------------------
data_folder = "data"
img_folder = "images"
mid_width = 5.1

# ---------------FILES ---------------------

df_stem = pd.read_csv(f"{data_folder}/2_df_stem_job_counts.csv")

df_educ_waffle = pd.read_csv(f"{data_folder}/df_educ_waffle.csv")
df_no_exp_waffle = pd.read_csv(f"{data_folder}/df_no_exp_waffle.csv")
df_relevance =  pd.read_csv(f"{data_folder}/df_relevance.csv")

bot_detail = df_relevance.logskills.quantile(0.33)
top_detail = df_relevance.logskills.quantile(0.66)
bot_relevance = df_relevance['%. of OJV skills'].quantile(0.33)
top_relevance = df_relevance['%. of OJV skills'].quantile(0.66)

df_comparable = df_relevance.query(f'`logskills` > {bot_detail}').query(f'`%. of OJV skills` > {bot_relevance}')
df_noncomparable = df_relevance[(df_relevance.logskills <= bot_detail) | (df_relevance['%. of OJV skills'] <= bot_relevance)]


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

def plot_stem_jobs(df_):
    fig = go.Figure(data =[
        go.Bar(name='STEM Jobs', x=df_stem['year'], y=df_stem['STEM'],
               marker_color='#30BBB1'),
        go.Bar(name='Non-STEM Jobs', x=df_stem['year'], y=df_stem['Non-STEM'],
               marker_color='lightgrey',)
        ])
    fig.update_layout(barmode='stack',
                      title='Jobs per year',
                      template='simple_white',
                      autosize=True,
                      xaxis=dict(type='category'),
                      xaxis_title='Year',
                      yaxis_title='Job Counts'
                      )
    fig.update_traces()
    st.plotly_chart(fig, use_container_width=True)

def plot_educ(df_educ_waffle):
    df_educ_waffle['Job Count'] = df_educ_waffle['Job Count'].apply(lambda x: int(x/100))

    # To plot the waffle Chart
    fig = plt.figure(
        FigureClass = Waffle,
        rows = 10,
        values = df_educ_waffle['Job Count'],
        labels = list(df_educ_waffle['Education']),
        legend = {'loc': 'lower left',
                  'bbox_to_anchor': (0,-.4),
                  'fontsize': 40
                 },
        colors = ['#d9d9d9', '#00c2cb', '#ffcd00'],
        figsize = (50, 80),
        dpi=100
    )

    st.pyplot(fig)


def plot_experience(df_no_exp_waffle):
    df_no_exp_waffle['Job Count'] = df_no_exp_waffle['Job Count'].apply(lambda x: int(x/100))
    
    # To plot the waffle Chart
    fig = plt.figure(
        FigureClass = Waffle,
        rows = 10,
        values = df_no_exp_waffle['Job Count'],
        labels = list(df_no_exp_waffle['Work Experience']),
        legend = {'loc': 'lower left',
                  'bbox_to_anchor': (0,-.4),
                  'fontsize': 40
                 },
        colors = ['#00c2cb', '#d9d9d9', '#ffcd00'],
        figsize = (50, 80),
        dpi=100
    )

    st.pyplot(fig)

def plot_curriculum_readiness():

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_comparable['%. of OJV skills'],
        y=df_comparable.logskills,
        name='Comparable',
        mode='markers',
        text=df_comparable['degree'],
        marker_color='#FDC13A' #'rgba(152, 0, 0, .8)'
    ))

    fig.add_trace(go.Scatter(
        x=df_noncomparable['%. of OJV skills'],
        y=df_noncomparable.logskills,
        name='Non-comparable',
        marker_color='Grey' #'rgba(255, 182, 193, .9)'
    ))

    # Set options common to all traces with fig.update_traces
    fig.update_traces(mode='markers', marker_size=10)
    fig.update_layout(title='Curriculum Readiness',
                      width=800, height=800,
                      yaxis_zeroline=False, xaxis_zeroline=False,
                      xaxis_title='% of Skills found in Job Postings',
                      yaxis_title='log(Number of Skills Identified)'
                      )


    st.plotly_chart(fig, use_container_width=True)

# -------------- LAYOUT --------------------
def get_contents():
    
    ##############
    ##  HEADER  ##
    ##############
    st.header("Branching from STEM Education")
    st.write("Correlating the Curricula and STEM Skills Necessary to Support In-Demand Jobs in the Philippines")

    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
    with row0_1:
        st.title('Data Overview')


    with row0_2:
        #st.image(f"{img_folder}/SLA_PH_Logo/Main-Horizontal.png", use_column_width=True)
        st.image(f"{img_folder}/SLA_PH_Logo/Black/Black-Horizontal.png", use_column_width=True)

        # if st.theme() == 'Dark':
        #     st.image(f"{img_folder}/SLA_PH_Logo/White/White-Horizontal.png", use_column_width=True)
        # else:
        #     st.image(f"{img_folder}/SLA_PH_Logo/Black/Black-Horizontal.png", use_column_width=True)
        #st.subheader('Streamlit App by [Tim Denzler](https://www.linkedin.com/in/tim-denzler/)')

    row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, mid_width, .1))
    with row1_1:
        h_text = open(f"{data_folder}/2_header.txt","r+")
        h_text = "".join(h_text.readlines())
        st.markdown(h_text)
        # st.markdown('There are three primary data types used for this project, all of which were from publicly available sources – existing labor taxonomies, online job vacancies, and government-issued curriculums.')

    ###############
    ##  CONTENT  ##
    ###############

    #--- STATS ---
    row2_spacer1, row2_1, row2_spacer2 = st.columns((.1, mid_width, .1))

    with row2_1:
        st.header("At a Glance")
        metrics_text = open(f"{data_folder}/2_metrics.txt","r+")
        metrics_text = "".join(metrics_text.readlines())
        st.markdown(metrics_text)


    row2a_spacer1, row2a_1,  row2a_2, row2a_spacer3, row2a_3, row2a_spacer4, row2a_4, row2a_spacer5 = st.columns((.2, 2, 3,.1,3,.1,2,.2))

    with row2a_1:
        st.subheader("Online Job Vacancies")
        st.metric('Job Posts', 56_150)
        st.metric('Job Titles (Normalized)', 4_402)
        st.metric('Occupations', 814)
        st.metric('STEM Occupations', 208)
        
    with row2a_2:
        st.text("")
        st.text("")
        st.text("")
        st.image(f"{img_folder}/monster.png", width=200)
        st.image(f"{img_folder}/glassdoor.jpg", width=200)
        st.image(f"{img_folder}/pinoyjobs.png", width=200)
        st.image(f"{img_folder}/cc.jpg", width=200)
        
    with row2a_3:
        st.subheader("Curriculums")
        st.metric('STEM Curriculums', 52)
        st.image(f"{img_folder}/ched.png", width=100)
        
    with row2a_4:
        st.subheader("Existing Labor Taxonomies (referenced)")
        st.image(f"{img_folder}/emsi-labor-market-analytics-vector-logo.png", width=100)
        st.image(f"{img_folder}/onet-circle.png", width=100)
    

    #--- Job Vacancies ---
    row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, mid_width, .1))
    with row3_1:
        st.header("Online Job Vacancies (OJV)")
        ojv_text = open(f"{data_folder}/2_ojv_text.txt","r+")
        ojv_text = "".join(ojv_text.readlines())
        st.markdown(ojv_text)
        
        ojv_more = open(f"{data_folder}/2_ojv_more.txt","r+")
        ojv_more = "".join(ojv_more.readlines())
        
        with st.expander("More on the Job Title Normalization Process"):
            st.markdown(ojv_more)

    row3a_spacer1, row3a_1, row3a_spacer2, row3a_2, row3a_spacer3  = st.columns((.2, 4, .1, 4, .2))
    with row3a_1:
        st.write('')
        st.markdown('Job Count Data:')
        st.write('')
        st.dataframe(df_stem.head(10))

    with row3a_2:
        plot_stem_jobs(df_stem)

    row3b_spacer1, row3b_1, row3b_spacer2, row3b_2, row3b_spacer2 = st.columns((.2, mid_width/2, .1, mid_width/2, .2))
    with row3b_1:

        st.subheader("Required Education for STEM jobs")
        educ_text = open(f"{data_folder}/2_educ_text.txt","r+")
        educ_text = "".join(educ_text.readlines())
        
        plot_educ(df_educ_waffle)
        st.markdown(educ_text)

    with row3b_2:
        st.subheader("Required Experience for STEM jobs")
        experience_text = open(f"{data_folder}/2_experience_text.txt","r+")
        experience_text = "".join(experience_text.readlines())

        plot_experience(df_no_exp_waffle)
        st.markdown(experience_text)


    #--- Curriculums ---
    row4_spacer1, row4_1, row4_spacer2 = st.columns((.1, mid_width, .1))
    with row4_1:
        st.header("Curriculums")
        curric_text = open(f"{data_folder}/2_curriculums.txt","r+")
        curric_text = "".join(curric_text.readlines())
        st.markdown(curric_text)

    row4a_spacer1, row4a_1, row4a_spacer2, row4a_2, row4a_spacer3  = st.columns((.2, 2, .1, 5, .2))

    with row4a_1:
        st.write('')
        st.write('')


    with row4a_2:
        plot_curriculum_readiness()
