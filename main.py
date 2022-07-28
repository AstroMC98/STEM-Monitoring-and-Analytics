import streamlit as st
from  PIL import Image
from streamlit_option_menu import option_menu
from st_aggrid import AgGrid
from unilab_pages.landing import get_contents as landing_content
from unilab_pages.emerging_jobs import get_contents as emerging_content
from unilab_pages.data_overview import get_contents as overview_content
from unilab_pages.compare import get_contents as compare_content
from unilab_pages.skill_analysis import get_contents as skill_content
from unilab_pages.faq import get_contents as faq_content

st.set_page_config(layout="wide")

with st.sidebar:
    st.sidebar.image(f"data/SLA_PH_Logo/Main-Square.png", use_column_width=True)
    choose = option_menu("Pages", ["Branching From STEM", "Data Overview", "Skill Analysis", "Emerging Jobs", "Curriculum Evaluation", "FAQs"],
                         icons=['house', 'boxes', 'diagram-3', 'graph-up-arrow','card-checklist', 'question-circle'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "#19486A", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#00c2cb"},
    }
    )

if choose == "Branching From STEM":
    landing_content()
elif choose == "Data Overview":
    overview_content()
elif choose == "Skill Analysis":
    skill_content()
elif choose == "Emerging Jobs":
    emerging_content()
elif choose == "Curriculum Evaluation":
    compare_content()
elif choose == "FAQs":
    faq_content()
