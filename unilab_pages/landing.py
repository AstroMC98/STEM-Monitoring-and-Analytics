import streamlit as st
from  PIL import Image
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid

logo = Image.open(r'images/logo.png')
icon = Image.open(r'images/icon.png')

def get_contents():
    col1_space1,col1_1, col1_2, col1_space1, = st.columns([0.15, 0.4, 0.3, 0.15])
    with col1_1:               # To display the header text using css style
        st.image(logo, width=130 )
    col2_space1, col2_1, col2_2, col2_space2 = st.columns([0.15, 0.4, 0.3, 0.15])
    with col2_1:
        st.markdown("""
        <style>
        @import url(https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@500&display=swap);

        .header {
        font-size:50px ;
        font-family: 'Libre Franklin', sans-serif;
        color: #000000;
        line-height: 1.4;
        }
        </style> """, unsafe_allow_html=True)
        st.markdown('<h1 class="header">Branching From</br>STEM Education</h1>', unsafe_allow_html=True)

        st.markdown("""
        <style>
        @import url(https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@300;500&display=swap);

        .thinFranklin {
        font-size:18px ;
        font-family: 'Libre Franklin', sans-serif;
        color: #000000;
        line-height: 1.4;
        }
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="thinFranklin">Correlating the Curriculums and STEM Skills Necessary to Support In-Demand Jobs within the Philippines</p>', unsafe_allow_html=True)

        st.markdown("""
        <style>
        .button {
          background-color: #04AA6D;
          border: none;
          color: white;
          padding: 20px;
          text-align: center;
          text-decoration: none;
          display: inline-block;
          font-size: 18px;
          margin: 2px 2px;
          font-weight: bold;
        }
        a:link{color:white;}
        a:visited{color:white;}
        a:hover{color:white;}
        a:focus{color:white;}
        a:active{color:white;}

        .button1 {border-radius: 2px;}
        .button2 {border-radius: 4px;}
        .button3 {border-radius: 8px;}
        .button4 {border-radius: 12px;}
        .button5 {border-radius: 50%;}
        </style> """, unsafe_allow_html=True)
        st.markdown('<a href="https://drive.google.com/file/d/1Q13vDqy4wxeH4Xw2IizTREKlhdH2raW3/view?usp=sharing" class="button button4">Link to Project Report</a>', unsafe_allow_html = True)
    with col2_2:               # To display brand logo
        st.image(icon, width = 500)
    with st.expander("See Project Rationale"):
        st.write(
     """
     Perhaps the best part of data science is seeing how data eventually transform into actionable insights on noble and meaningful applications. This project, led by the Asian Institute of Management Masters in Data Science Program (AIM) together with the Unilab Foundation and the STEM Leadership Alliance PH, is one example of how we can leverage data science to understand relevant situations and hopefully produce data-driven insights that can be the catalyst of change needed to make the world, albeit only in a specific or limited aspect, a little better. Specifically, we wanted to identify the emerging STEM-related jobs in the Philippines, know which STEM skills these jobs require, and assess how well our current educational system prepares our graduates for the most in-demand STEM jobs.

     In the Philippines, it is common to associate quality education with a decent job. After all, it is understandable to think that educational programs supposedly equip graduates with the skills they need to be competent in the job market. However, the needs of employers change with the evolution of technology and other factors such as economic conditions and market trends. This is contrary to curriculums which tend to be static rather than adaptive. Thus, there is a gap between the skills graduates acquire in school and what the job market expects them to have.

     This challenge has been the inspiration and the center of our project. In attempting to answer "How do we assure that STEM graduates have the skills for the Job Market of Today?", we implemented a 5-part pipeline: Extract, View, Review, Compare, and Adopt. It has a few limitations, particularly in the Extract part. First and foremost, the data is limited to what was obtainable in our implementation. Second, since there are no standard skills and job taxonomy specific to the Philippines, we used O*NET, a taxonomy based on the US labor market. As such, the data we have might lack a local context. Despite this, we find the project results valuable, and we are excited to have this improved with better data in the future.

     In each of the first four parts, we have detailed its importance and the techniques and methods we implemented in a report. The hope is that for the fifth and final part of our pipeline, our dashboard containing the data-driven results, transformed into a digestible bit of information and visualizations, can act as a decision support tool for the Unilab Foundation, the STEM Leadership Alliance PH, and other concerned educational institutions to turn our study into actionable insights through policies.
     """)
