
# --------------- IMPORTS ------------------
import streamlit as st
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import altair as alt
from sklearn.preprocessing import StandardScaler
import statistics
import pandas as pd
import numpy as np
import pickle
from io import BytesIO
from ast import literal_eval
import glob

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import spatial

# ---------------- SETUP -------------------
data_folder = "data/"
img_folder = "images/"
mid_width = 5.1

finalvalues = pd.read_csv('data/allvalues.csv')
allprofiles = pd.read_csv('data/allprofiles.csv')
top50_jobs = pd.read_csv('data/top50_occupation_stem.csv')['occupation'].to_list()
top50_jobs = top50_jobs[:10]

df_jaccard = pd.read_csv("data/df_jaccard.csv")
rename_occ = {'Electrical And Electronic Engineering Technologists And Technicians':
              'Electrical And Electronic Engineering'}
df_jaccard = df_jaccard.rename(columns={'Unnamed: 0':'emerging_occ'}).set_index('emerging_occ')
df_jaccard = df_jaccard.round(2)

df_profiles = pd.read_csv(f"{data_folder}/emerging_job_skill_profiles.csv")

cat = ['Information Techology & Systems Administration','Hard Sciences & Engineering','Data Analytics & Data Science','Advertising, Marketing, & Design','Software Developers & Programmers','Biological Sciences, Healthcare, Brand Management','Accounting, Logistics, & Business Administration','Information Techology & Systems Administration']

df_all = pd.read_csv('data/df_all.csv')

non_stem = ['Criminology', 'Development Communications', 'Business Administration',
            'Social Work Program', 'Accountancy', 'Management Accounting', 'Accounting Information System',
            'Internal Auditing','Architecture', 'Hospitality Tourism Management']

df_curr = pd.read_csv("data/df_curr_final.csv")
df_curr['All Hard Skills'] = df_curr['All Hard Skills'].apply(lambda x: eval(x))
df_curr = df_curr[~df_curr['degree'].isin(non_stem)]

with open('data/jobsandcurric.pickle', 'rb') as handle:
    jobcurric = pickle.load(handle)

with open(f'data/Word2Vec/Word2Vec_500_-0.5.pkl', 'rb') as handle:
    w2v_model = pickle.load(handle)


df_emerging = pd.read_csv('data/df_emerging.csv')
df_emerging['Central Nodes'] = [literal_eval(x) for x in df_emerging['Central Nodes']]

df_jobs = pd.read_csv(f"{data_folder}emerging_job_skill_profiles.csv")
df_jobs['Core Skills'] = df_jobs['Core Skills'].apply(lambda x: eval(x))
df_jobs['Specialized Skills'] = df_jobs['Specialized Skills'].apply(lambda x: eval(x))
df_jobs['Adjacent Skills'] = df_jobs['Adjacent Skills'].apply(lambda x: eval(x))
df_jobs['Bridging Skills'] = df_jobs['Bridging Skills'].apply(lambda x: eval(x))
df_jobs['All Skills'] = df_jobs.iloc[:, 1:].sum(axis=1)

df_mean = pd.read_csv("data/df_mean.csv")
df_mean = df_mean.rename(columns={'Unnamed: 0':'emerging_occ'}).set_index('emerging_occ')
df_mean = df_mean.round(2)

ready_curric_list = ['Aeronautical Engineering','Agricultural and Biosystems Engineering',
                     'Agriculture','Chemical Engineering','Civil Engineering','Computer Engineering',
                     'Computer Science','Electrical Engineering','Electronics Engineering','Exercise and Sports Science',
                     'Food Technology','Industrial Engineering','Information Systems','Information Technology',
                     'Materials Engineering','Mathematics','Mechanical Engineering','Metallurgical Engineering',
                     'Mining Engineering','Nutrition and Dietetics','Psychology','Radiologic Technology','Speech Language Pathology']
other_curric_list = list(set(df_curr.degree.to_list()) - set(ready_curric_list))
sorted_columns = ['Occupation']+ready_curric_list+other_curric_list

@st.cache
def convert_df(df):
   return df.to_csv().encode('utf-8')

def mean_embeddings(skills):
    embeddings = []
    for skill in skills:
        embedding = w2v_model.wv[skill]
        embeddings.append(embedding)
    return np.mean(embeddings,axis = 0)

def csv_job_profile(jobs_comparison):
    """Returns CSV for checklist template of the specific job provided"""

    my_data = df_profiles[df_profiles.Occupation == jobs_comparison]
    core_list = literal_eval(my_data.iloc[0,1])
    spec_list = literal_eval(my_data.iloc[0,2])
    adj_list = literal_eval(my_data.iloc[0,3])
    role_list = literal_eval(my_data.iloc[0,4])

    all_skills = core_list + spec_list + adj_list + role_list
    skill_type = ['core']*len(core_list) + ['specialized']*len(spec_list)+['adjacent']*len(adj_list)+['role-based']*len(role_list)

    df_output = pd.DataFrame({
        'Skill': all_skills,
        'Category': skill_type,
        'Checklist': ['']*len(skill_type)
        })

    return convert_df(df_output)

def combine_skills(x):
    return x[:1]

df_jobs['Core10'] = df_jobs['Core Skills'].apply(lambda x: x[:10] if len(x)>10 else x)
df_jobs['Spec10'] = df_jobs['Specialized Skills'].apply(lambda x: x[:10] if len(x)>10 else x)
df_jobs['Adj10'] = df_jobs['Adjacent Skills'].apply(lambda x: x[:10] if len(x)>10 else x)
df_jobs['Role10'] = df_jobs['Bridging Skills'].apply(lambda x: x[:10] if len(x)>10 else x)
df_jobs['Top10_combined'] = df_jobs.iloc[:, 6:10].sum(axis=1)
df_jobs['combined_len'] = df_jobs['Top10_combined'].apply(len)


def radial_preprocess(jobs):
    values = finalvalues[finalvalues['Jobs']==jobs].T.loc['Clusters'].values.flatten().tolist()
    values += values[:1]
    return values

def create_radar(jobs):
    values = radial_preprocess(jobs)
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]*1]*1)
    fig.add_trace(go.Scatterpolar(
          name = f"{jobs}",
          r = values,
          theta = cat,
        ), 1, 1)
    fig.update_traces(fill='toself')
    fig.update_layout(polar=dict(radialaxis_angle=0, angularaxis = dict(rotation=90, direction='clockwise'), radialaxis = dict(tickvals=[0.1, 0.2,0.3,0.4,0.5])))
    st.plotly_chart(fig, use_container_width=True)

def create_2radar(jobs):
    categories=list(range(1,8))
    value1 = radial_preprocess(jobs[0])
    value2 = radial_preprocess(jobs[1])


    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'polar'}]*1]*1)
    fig.add_trace(go.Scatterpolar(
          name = f"{jobs[0]}",
          r = value1,
          theta = cat,
        ), 1, 1)
    fig.add_trace(go.Scatterpolar(
          name = f"{jobs[1]}",
          r = value2,
          theta = cat,
        ), 1, 1)
    fig.update_traces(fill='toself')
    fig.update_layout(polar=dict(radialaxis_angle=0,
                                 angularaxis = dict(rotation=90,
                                                    direction='clockwise'),
                                 radialaxis = dict(tickvals=[0.1, 0.2,0.3,0.4,0.5])))
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=-0.2,
    xanchor="center",
    x=0.5
))
    st.plotly_chart(fig, use_container_width=True)

def create_profiles(jobs):

    df = allprofiles[allprofiles['job']==jobs]
    core_x = df[df['skill profile']=='core']['standard_degree']
    core_y = df[df['skill profile']=='core']['standard_count']

    adjacent_x = df[df['skill profile']=='adjacent']['standard_degree']
    adjacent_y = df[df['skill profile']=='adjacent']['standard_count']

    specialized_x = df[df['skill profile']=='specialized']['standard_degree']
    specialized_y = df[df['skill profile']=='specialized']['standard_count']

    bridging_x = df[df['skill profile']=='bridging']['standard_degree']
    bridging_y = df[df['skill profile']=='bridging']['standard_count']
    fig = go.Figure()

    # Add traces
    fig.add_trace(go.Scatter(x=core_x, y=core_y, text =df[df['skill profile']=='core']['skill'],
                    mode='markers',
                    name='Core Skills'))
    fig.add_trace(go.Scatter(x=adjacent_x, y=adjacent_y, text =df[df['skill profile']=='adjacent']['skill'],
                mode='markers',
                name='Adjacent Skills'))
    fig.add_trace(go.Scatter(x=specialized_x, y=specialized_y, text =df[df['skill profile']=='specialized']['skill'],
                mode='markers',
                name='Specialized Skills'))
    fig.add_trace(go.Scatter(x=bridging_x, y=bridging_y, text =df[df['skill profile']=='bridging']['skill'],
                    mode='markers',
                    name='Role-Specific Skills'))
    fig.update_xaxes(title_text='Degree Centrality')
    fig.update_yaxes(title_text='Job Occurence')
#     fig.update_layout(template='simple_white', autosize=False, width=500, height=500)
#     fig['layout'].update(scene=dict(aspectmode="data"))
#     fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
    st.plotly_chart(fig, use_container_width=True)



def prep_ribbon_skill_degree(job, my_degrees):
    import plotly.graph_objects as go

    df_ds_curric = df_all[df_all.Occupation==job].T.drop(['Occupation', 'All Skills'])[1:]
    df_ds_curric.columns=['skills']
    df_ds_curric['skills'] = df_ds_curric['skills'].apply(lambda x:literal_eval(x))
    df_ds_curric = df_ds_curric.explode("skills").dropna().reset_index()
    df_ds_curric.columns=['degree', 'skill']

    #print("all degrees: ", list(df_ds_curric.degree.unique()))

    all_curriculums = ready_curric_list+['Nursing', 'Physics', 'Statistics']

    # Add in Uploads



    #print("focus degrees:", my_degrees)
    df_ds_curric = df_ds_curric[df_ds_curric.degree.isin(my_degrees)].copy()

    curr_skills = list(df_ds_curric.skill.unique())

    #Create skill group index map
    this_skill_map = dict(zip(curr_skills, range(0,len(curr_skills))))
    n = len(this_skill_map) # previous number

    this_curric_map = dict(zip(list(df_ds_curric.degree.unique()),
                               range(n,n+len(all_curriculums))))

    df_ds_curric['source'] = df_ds_curric['skill'].map(this_skill_map)
    df_ds_curric['target'] = df_ds_curric['degree'].map(this_curric_map)

    # Create Skill Group index map
    this_skill_map = dict(zip(curr_skills, range(0,len(curr_skills))))
    n = len(this_skill_map) # previous number

    # Create Curriculum map
    all_curriculums = ready_curric_list+['Nursing', 'Physics', 'Statistics']

    this_curric_map = dict(zip(list(df_ds_curric.degree.unique()),
                               range(n,n+len(all_curriculums))))

    # Map it back to the jobs and curriculums
    df_ds_curric['source'] = df_ds_curric['skill'].map(this_skill_map)
    df_ds_curric['target'] = df_ds_curric['degree'].map(this_curric_map)

    label = (list(this_skill_map.keys()) + list(this_curric_map.keys()))

    source = df_ds_curric['source'].to_list()
    # print(source)

    target = df_ds_curric['target'].to_list()
    # print(target)


    return label, source, target

def plot_ribbon_skill_degree(label, source, target,jobs_comparison,
                             link_color='#FCC308', node_color="#EC7063",
                             height=700, width=1000):

    color_node = ['navy']
    this_core = []
    for i in label:
        if i in this_core:
            color_node.append(node_color)
    color_link = []
    highlight_curric = []

    for i in target:
        if label[i] in highlight_curric:
            color_link.append(link_color)
        else:
            color_link.append('#E4E2E2')

    value = [1]*len(source) # data to dict, dict to sankey
    link = dict(source = source, target = target, value = value, color=color_link) #color='#E4E2E2')
    node = dict(label = label, pad=50, thickness=5, color=color_node) #'navy')
    data = go.Sankey(link = link, node=node)# plot

    layout = go.Layout(
        autosize=False,
        width=width,
        height=height,

        xaxis= go.layout.XAxis(linecolor = 'grey',
                              linewidth = 1,
                              mirror = True),

        yaxis= go.layout.YAxis(linecolor = 'grey',
                              linewidth = 1,
                              mirror = True),

        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad = 4
        )
    )

    fig = go.Figure(data, layout=layout)
    # fig.update_traces(orientation="v", selector=dict(type='sankey'))
    fig.update_traces(arrangement="freeform",  selector=dict(type='sankey'))
    fig.update_layout(title_text=f"Sankey: {jobs_comparison}", font_size=15)
    st.plotly_chart(fig, use_container_width=True)


def create_jaccard(df_jaccard):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax = sns.heatmap(df_jaccard.T.rename(columns=rename_occ),
                 cmap="YlGnBu",
                 annot=True,
                 vmax=1, vmin=0
                )


    plt.ylabel('Skills', labelpad=20)
    plt.xlabel('Emerging Occupations', labelpad=20)
    plt.title("Jaccard Similarity", pad=20)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

def create_cej(df, jobFilter, metric):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)

    df = df.copy()

    #Add in uploads
    if metric == 'Cosine Similarity':
        for uploaded_file_name in glob.glob('data/uploads/*.csv'):
            jobName = uploaded_file_name.split('/')[-1].split('\\')[1][:-4]
            df_uploaded = pd.read_csv(uploaded_file_name, index_col = 0)
            df_uploaded.dropna(inplace = True)
            all_skills = ['_'.join(x.split()) for x in df_uploaded.Skill]
            s2 = mean_embeddings(all_skills)
            scores = []
            for i,s1_skills in enumerate(df_emerging['Central Nodes'].tolist()):
                s1 = mean_embeddings(s1_skills)
                sim_score = 1 - spatial.distance.cosine(s1,s2)
                scores.append(sim_score)
            df[jobName] = scores


    elif metric == 'Jaccard Similarity':
        for uploaded_file_name in glob.glob('data/uploads/*.csv'):
            jobName = uploaded_file_name.split('/')[-1].split('\\')[1][:-4]
            df_uploaded = pd.read_csv(uploaded_file_name, index_col = 0)
            df_uploaded.dropna(inplace = True)
            all_skills = ['_'.join(x.split()) for x in df_uploaded.Skill]
            jccrd = []
            for i,s1_skills in enumerate(df_emerging['Central Nodes'].tolist()):
                intrsct = list(set(s1_skills) & set(all_skills))
                unn = list(set(s1_skills).union(all_skills))
                j_score = len(intrsct)/len(unn)
                jccrd.append(j_score)
            df[jobName] = jccrd

    ax = sns.barplot(x = df.T[jobFilter],
                     y = df.T.index,
                     palette = "flare")

    plt.ylabel('Emerging Occupations', labelpad=20)
    plt.xlabel('Similarity', labelpad=20)
    plt.title(metric, pad=20)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

def create_cosine1(df_mean):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax = sns.heatmap(df_mean.T.rename(columns=rename_occ),
                 cmap="Spectral",
                 annot=True,
                 vmax=1, vmin=0
                )


    plt.ylabel('Skills', labelpad=20)
    plt.xlabel('Emerging Occupations', labelpad=20)
    plt.title("Mean Cosine Similarity", pad=20)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

def create_cosine2(df_sum):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
    ax = sns.heatmap(df_sum.T.rename(columns=rename_occ),
                 cmap="Spectral",
                 annot=True,
                 vmax=1, vmin=0
                )


    plt.ylabel('Skills', labelpad=20)
    plt.xlabel('Emerging Occupations', labelpad=20)
    plt.title("Sum Cosine Similarity", pad=20)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)



# -------------- LAYOUT --------------------
def get_contents():

    ###############
    ##  CONTENT  ##
    ###############
    row0_spacer1, row0_1, row0_spacer2 = st.columns((.1, 5.1, .2))
    with row0_1:
        st.header("Curriculum Evaluation")
        st.write("In this page, we visualize and explore the skill gap between emerging jobs and STEM-related courses through the use of distance metrics such as Jaccard Similarity and Cosine Similarity.",
                 "This page also showcases the skill profile of each emerging job with respect to the current taxonomy that we have created from the data that we were able to collate."
                )

    row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 5.1, .2))
    with row1_1:
        st.header('Proportion of skills per cluster')

    row2_spacer1, row2_1L , _, row2_1R, row2_spacer2 = st.columns((.1, 3.5, 0.0125, 2.5, .2))
    with row2_1R:

        radar_options = list(top50_jobs[:10])
        job_radar1 = st.selectbox(
        label = 'Select an Emerging Job to View the it\'s Job Profile:',
        options = radar_options,
        index = 0
        )

        radar_options.remove(job_radar1)
        job_radar2 = st.selectbox(
        'Select an Emerging Job to Compare and View the it\'s Job Profile:',
        options = radar_options,
        index = 0
        )

        st.write('The Radar plot/s below show the proportion of skills per skill cluster per emerging job. ',
                 'If you hover on the points you can see the exact proportion of each cluster in a specific emerging job. ',
                 'For example in the radar plot below, 43.75% of the skills extracted for Robotics Engineers are Software Developers & Programmers, 15.63% are Accounting, Logistics, & Business Administration, 12.5% are Biological Sciences, Healthcare, and Brand Management.  If 2 jobs were chosen, a comparison of the proportion of skills per cluster per job will be shown.')

    with row2_1L:
        create_2radar([job_radar1, job_radar2])

    row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 5.1, .2))
    with row3_1:
        st.header('Skill Profiles per Job')
        jobs_profile = st.selectbox("Choose an emerging job to view the skill profiles.", top50_jobs[:10])
        create_profiles(jobs_profile)

    row4_spacer1, row4_1L, _, row4_1R, row4_spacer2 = st.columns((0.50, 0.75, 0.125, 3.5, 1))
    with row4_1L:
        st.download_button(
           "Download Job Profile",
           csv_job_profile(jobs_profile),
           f"{jobs_profile}_[UPLOADED]_[SCHOOL or ORG].csv",
           "text/csv",
           key='download-csv'
        )
    with row4_1R:
        st.markdown('The scatter plot below shows the Skill Profile of each skill present in a specific emerging job. The x-axis is the standardized degree centrality and the y-axis is the standardized frequency. The higher the degree centrality, the more connected a skill is while higher frequency means a skills is mentioned more than the others.')

    row5_spacer1, row5_1, row5_spacer2 = st.columns((.1, 5.1, .2))
    with row5_1:
        st.header('Uploading New Curriculum Data')
        uploaded_files = st.file_uploader("Choose a CSV file", type = 'csv', accept_multiple_files=True)
        for uploaded_file in uploaded_files:
             bytes_data = uploaded_file.read()
             df_upload = pd.read_csv(BytesIO(bytes_data), index_col = 0)
             df_upload.to_csv(f'data/uploads/{uploaded_file.name}.csv')
             st.write("Uploaded filename:", uploaded_file.name)
             st.write(bytes_data)

    row6_spacer1, row6_1, row6_spacer2 = st.columns((.1, 5.1, .2))
    with row6_1:
        st.header('Comparing Emerging Jobs')
        st.write('The next two graphs will show us similarity scores where the closer the similarity score to 1.0 is, means that the curriculum and the emerging job are similar.\n')

    row7_spacer1, row7_1L, _, row7_1R, row7_spacer2 = st.columns((.1, 5, 0.125, 3, .2))
    with row7_1R:

        cej_select1 = st.selectbox(
        label = 'Type of Metric:',
        options = ['Jaccard Similarity', 'Cosine Similarity'],
        index = 0
        )

        cej_select2 = st.selectbox(
        label = 'Emerging Job To Compare:',
        options = list(top50_jobs[:10]),
        index = 0
        )

        st.write('The next two graphs will show us similarity scores where the closer the similarity score to 1.0 is, means that the curriculum and the emerging job are similar.\n')

    with row7_1L:
        if cej_select1 == 'Jaccard Similarity':
            create_cej(df_jaccard, cej_select2, cej_select1)
        elif cej_select1 == 'Cosine Similarity':
            create_cej(df_mean, cej_select2, cej_select1)

    row8_spacer1, row8_1, row8_spacer2 = st.columns((.1, 5.1, .2))
    with row8_1:
        st.header('Summary Visualizations Heatmaps')
        st.write('The next two graphs will show us similarity scores where the closer the similarity score to 1.0 is, means that the curriculum and the emerging job are similar.\n')

    row9_spacer1, row9_1L, _, row9_1R, row9_spacer2 = st.columns((.1, 3, 0.125, 3, .2))
    with row9_1L:
        create_jaccard(df_jaccard)

    with row9_1R:
        create_cosine1(df_mean)

    row10_spacer1, row10_1, row10_spacer2 = st.columns((.1, 5.1, .2))
    with row10_1:
        st.header('Job vs Curriculum')
        st.write('The ribbon chart below shows the skills visible in both emerging job and curriculums.')
        jobs_comparison = st.selectbox("Choose an emerging job to compare with curriculum(s).", top50_jobs[:10])
        curriculums = st.multiselect("Choose a list of curriculum(s) to compare.", jobcurric[jobs_comparison], default = jobcurric[jobs_comparison][:5])
        label, source, target = prep_ribbon_skill_degree(jobs_comparison, curriculums)
        plot_ribbon_skill_degree(label, source, target,
                             jobs_comparison,
    #                          link_color="#73E4DC",
                             node_color="#EC7063",
                             height = 500, width=700)
