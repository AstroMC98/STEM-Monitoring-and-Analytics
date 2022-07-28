# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 00:04:17 2022
@author: Sarah Isabel Mendoza
"""
# --------------- IMPORTS ------------------
import streamlit as st
import pandas as pd

# ---------------- SETUP -------------------
data_folder = "data"
img_folder = "images"
mid_width = 5.1

# ---------------FILES ---------------------




# -------------- HELPER FUNCTIONS ----------


# -------------- LAYOUT --------------------
#############
## SIDEBAR ##
#############


###############
##  CONTENT  ##
###############
def get_contents():
    # st.header("Branching from STEM Education")
    # st.write("Correlating the Curricula and STEM Skills Necessary to Support In-Demand Jobs in the Philippines")

    row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
    with row0_1:
        st.title('Frequently Asked Questions (FAQs)')

    df_faq = pd.read_excel(f"{data_folder}/faqs.xlsx", engine='openpyxl').fillna('')
    # st.dataframe(df_faq)

    for c in df_faq.Category.unique():
        with st.expander(c):
            for each in df_faq[df_faq.Category == c].iterrows():
                # st.dataframe(each)
                st.markdown('**'+ each[1]['Question'] +'**')
                st.markdown(each[1]['Answer'])
                if each[1]['Image_file'] != "":
                    st.image(f"{img_folder}/faq/{each[1]['Image_file']}")

                st.write('')

    # with st.expander("Navigation"):
    #     for each in df_faq[df_faq.Category == 'Navigation'].iterrows():
    #         # st.dataframe(each)
    #         st.markdown('**'+ each[1]['Question'] +'**')
    #         st.markdown(each[1]['Answer'])

    # with st.expander("Data"):
    #     st.markdown('hello')

    # with st.expander("Methodology"):
    #     st.markdown('hello')
