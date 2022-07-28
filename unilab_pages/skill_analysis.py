import streamlit as st
from  PIL import Image
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

# Additional Packages
import networkx as nx
import pandas as pd
import pickle
from itertools import combinations
from scipy import sparse
from ast import literal_eval
from functools import reduce

import re
import numpy as np
import itertools
import pandas as pd
import glob
import plotly.graph_objects as go
from collections import Counter
import seaborn as sns
from streamlit_plotly_events import plotly_events

from st_aggrid import AgGrid
pd.options.plotting.backend = "plotly"

#Setup
data_folder = 'data'
img_folder = "images/"
mid_width = 5.1

job_networks = ['Blockchain_Engineers','Business_Intelligence_Analysts', 'Computer_And_Information_Systems_Managers',
                'Computer_Network_Support_Specialists', 'Computer_Systems_Analysts', 'Computer_Systems_Engineers_Architects',
                'Data_Scientists', 'Electrical_And_Electronic_Engineering_Technologists_And_Technicians',
                'Industrial-Organizational_Psychologists', 'Information_Security_Engineers', 'Medical_Records_Specialists',
                'Operations_Research_Analysts', 'Robotics_Engineers', 'Software_Developers', 'Software_Quality_Assurance_Analysts_And_Testers']


# Load Vague Skills
text_file = open(f"{data_folder}/Skills/vague_skills.txt", "r")
vague_skills = text_file.readlines()
vague_skills = [x.replace('\n','') for x in vague_skills]

# Load OJV
df_ojv = pd.read_csv(f'{data_folder}/df_ojv_HARD.csv')
df_ojv['All Hard Skills'] = df_ojv['All Hard Skills'].apply(literal_eval)
df_ojv.dropna(subset = ['job_title'], axis = 'rows', inplace = True)
df_ojv['All Hard Skills'] = df_ojv['All Hard Skills'].apply(lambda x: [s for s in x if s != None])

# Map Normalized Job Titles
normalized_titles = pd.read_csv(f'{data_folder}/df_id_normalized_stem_groups.csv')
normalized_titles = normalized_titles.query('`similarity` > 0.2')
id_to_NormOccu = dict(zip(normalized_titles.id, normalized_titles['Onet Occupation']))
df_ojv['NormOcc'] = df_ojv.id.apply(lambda x: id_to_NormOccu.get(x, 'Unmatched Job Titles'))

# Load Word2Vec model
with open(f'{data_folder}/Word2Vec/Word2Vec_500_-0.5.pkl', 'rb') as handle:
    w2v_model = pickle.load(handle)

def load_network(file):
    data_matrix = pd.read_csv(f'{data_folder}/networks/Job_title_{file}_skill_cooccurence_df.csv',index_col = 0)
    Gmain = nx.from_pandas_adjacency(data_matrix)
    sim_dct ={}
    for n1,n2 in Gmain.edges():
        sim_dct[(n1,n2)] = {'c':w2v_model.wv.similarity(n1,n2)}
    nx.set_edge_attributes(Gmain, sim_dct)

    def filter_edge(n1, n2):
        return Gmain[n1][n2]['c'] > 0

    G_course = nx.subgraph_view(Gmain, filter_edge=filter_edge).copy()
    Gcc = sorted(nx.connected_components(G_course), key=len, reverse=True)
    G_course = G_course.subgraph(Gcc[0])
    return G_course

def generate_network(filename):

    G = load_network(filename)
    labels = pd.read_csv(f'{data_folder}/labels/Job_title_{filename}_labels.csv')
    label_dct = dict(zip(labels.node, labels.NMF_label))
    H = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = H[edge[0]] #G.nodes[edge[0]]['pos']
        x1, y1 = H[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    cluster = []
    for node in G.nodes():
        if G.degree[node] == 0:
            continue
        x, y = H[node]
        cluster.append(label_dct[node])
        node_x.append(x)
        node_y.append(y)

    network_df = pd.DataFrame({'x':node_x, 'y':node_y, 'cluster':cluster})

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            #reversescale=True,
            color=cluster,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Cluster Membership',
                xanchor='left',
                titleside='right'
            ),
            line_width=2)
            )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        #node_adjacencies.append(len(adjacencies[1]))
        node_adjacencies.append(label_dct[adjacencies[0]])
        node_text.append(f'Skill : {adjacencies[0]}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                # annotations=[ dict(
                #     text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                #     showarrow=False,
                #     xref="paper", yref="paper",
                #     x=0.005, y=-0.002 ) ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    st.plotly_chart(fig)

# -------------- LAYOUT --------------------
##############
##  HEADER  ##
##############
def get_contents():
    st.header("Branching from STEM Education")
    st.write("Correlating the Curricula and STEM Skills Necessary to Support In-Demand Jobs in the Philippines")

    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
    with row0_1:
        st.title('Skill Networks of the Emerging Jobs within the Philippines')


    with row0_2:
        #st.image(f"{img_folder}/SLA_PH_Logo/Main-Horizontal.png", use_column_width=True)
        st.image(f"{img_folder}/SLA_PH_Logo/Black/Black-Horizontal.png", use_column_width=True)

    row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, mid_width, .1))
    with row1_1:
        network_select = st.selectbox('Select a Job Network to Explore:',
                                      job_networks, 6
        )
        generate_network(network_select)

    row4_spacer1, row4_1, row4_spacer2 = st.columns((.1, 5.1, .2))
    with row4_1:
        st.header('Skill Statistics')
        st.subheader('A view of the most similar skills and the most relevant jobs to a selected skill')

        df_skills = df_ojv[['id', 'datePosted', 'NormOcc', 'All Hard Skills']]
        #df_skills_exploded = df_skills.explode(['All Hard Skills'])
        df_skills.rename({'id':'Job ID', 'NormOcc' : 'Job Title', 'All Hard Skills': 'Skills'}, axis = 1, inplace = True)
        df_skills_exploded = df_skills.explode('Skills')

        AgGrid(df_skills)

        skill_selected = st.text_input('Skill Phrase or Skill Word to Analyze', 'NumPy')

    row5_spacer1, row5_1, row5_spacer_between, row5_2 ,row5_spacer2 = st.columns((.1, 2.5,.1, 2.6, .2))
    with row5_1:
        st.subheader(f'Top Job Titles of {skill_selected}')
        df_skills_exploded_match = df_skills_exploded.query(f'`Skills` == "{skill_selected}"')
        AgGrid(df_skills_exploded_match['Job Title'].value_counts().reset_index())

    with row5_2:
        st.subheader(f'Most Similar or Most Relevant Skills to {skill_selected}')
        most_sim = pd.DataFrame(w2v_model.wv.most_similar([skill_selected], topn=10))
        most_sim.rename({0:'Skill', 1:'Similarity/Relevance'}, axis = 1, inplace = True)
        print(most_sim)
        AgGrid(most_sim)

    row6_spacer1, row6_1, row6_spacer2 = st.columns((.1, 5.1, 2))
    with row6_1:
        st.subheader(f'Trend of Usage of {skill_selected} over the years')
        df_skills_exploded_match['year'] = df_skills_exploded_match['datePosted'].apply(lambda x: x.split('-')[0])
        df_skills_exploded_trend = df_skills_exploded_match.groupby(['year'])['Skills'].agg('count').reset_index().sort_values(['year'])
        print(df_skills_exploded_trend)
        fig = px.line(df_skills_exploded_trend, x="year", y="Skills")
        st.plotly_chart(fig)
