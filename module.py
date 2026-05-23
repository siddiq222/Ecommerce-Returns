import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import base64
import io
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import power_transform
import warnings
warnings.filterwarnings("ignore")
import json


with open("eda_insights.json", "r") as f:
    data = json.load(f)

def add_bg_image(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            /* Full screen background */
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}

            /* Dark overlay for readability */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0; left: 0;
                width: 100%; height: 100%;
                background: rgba(0, 0, 0, 0.55);
                z-index: 0;
            }}

            /* Keep content above overlay */
            .stApp > * {{
                position: relative;
                z-index: 1;
            }}

            /* Make main block transparent */
            .block-container {{
                background: transparent !important;
            }}

            /* Sidebar background */
            section[data-testid="stSidebar"] {{
                background: rgba(0, 0, 0, 0.6) !important;
            }}

            /* White text for visibility */
            h1, h2, h3, p, label, .stMarkdown {{
                color: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

def fun(df):
    st.markdown("""
            <style>
            [data-testid="stSelectbox"] select,
            [data-testid="stSelectbox"] > div > div {
                background-color: rgba(255, 255, 255, 0.15) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.4) !important;
                border-radius: 6px !important;
            }
            
            /* ── Number Input ── */
            [data-testid="stNumberInput"] input {
                background-color: rgba(255, 255, 255, 0.15) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.4) !important;
                border-radius: 6px !important;
            }

            /* +/- buttons */
            [data-testid="stNumberInput"] button {
               background-color: rgba(255, 255, 255, 0.15) !important;
               color: white !important;
               border: 1px solid rgba(255, 255, 255, 0.4) !important;
            }

            [data-testid="stNumberInput"] button:hover {
                background-color: #f0f0f0 !important;
            }
            </style>
        """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    /* Broader selector for dropdown portal */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] li {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
    }
                
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #f0f0f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    info=st.radio('pick one',['shape','info','describe','head'],horizontal=True)
    if info=='shape':
        rows, cols = df.shape
        st.metric(label="Dataset Shape", value=f"{rows:,} rows × {cols} cols")
    elif info=='info':
        info_df=pd.DataFrame({"Column": df.columns,
                            "Non-Null Count": df.notnull().sum().values,
                            "Null Count": df.isnull().sum().values,
                            "Dtype": df.dtypes.values,
                            "Unique Values": df.nunique().values})
        st.table(info_df)  
    elif info=='describe':
        nc=st.selectbox('pick data type',['numerical','categorical'])
        if nc=='numerical':
            html_table = df.describe().to_html()
            st.markdown(f"""
                <div style="
                    overflow-x: auto;
                    overflow-y: auto;
                    max-height: 350px;
                    background: rgba(255,255,255,0.1);
                    border-radius: 8px;
                    padding: 10px;
                ">
                    <style>
                        .custom-table {{ border-collapse: collapse; width: 100%; }}
                        .custom-table th, .custom-table td {{
                            padding: 8px 14px;
                            color: white;
                            border: 1px solid rgba(255,255,255,0.2);
                            white-space: nowrap;
                        }}
                        .custom-table th {{ background: rgba(255,255,255,0.15); }}
                        .custom-table tr:hover td {{ background: rgba(255,255,255,0.1); }}
                    </style>
                    {html_table.replace('<table', '<table class="custom-table"')}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.table(df.describe(include='object'))
    else:
        html_table = df.head(10).to_html()
        st.markdown(f"""
            <div style="
                overflow-x: auto;
                overflow-y: auto;
                max-height: 350px;
                background: rgba(255,255,255,0.1);
                border-radius: 8px;
                padding: 10px;
            ">
                <style>
                    .custom-table {{ border-collapse: collapse; width: 100%; }}
                    .custom-table th, .custom-table td {{
                        padding: 8px 14px;
                        color: white;
                        border: 1px solid rgba(255,255,255,0.2);
                        white-space: nowrap;
                    }}
                    .custom-table th {{ background: rgba(255,255,255,0.15); }}
                    .custom-table tr:hover td {{ background: rgba(255,255,255,0.1); }}
                </style>
                {html_table.replace('<table', '<table class="custom-table"')}
            </div>
        """, unsafe_allow_html=True)
def analysis(df):
    type=st.selectbox('pick one type of Analysis',['Univariate Analysis','Bivariate Analysis','Multivariate Analysis'])
    st.subheader(type)
    num_col=list(df.iloc[:,:9].columns)
    cat_col=list(df.iloc[:,9:].columns)
    num_col2=['customer_age','past_return_rate','delivery_delay_days']
    def univariate(col):
        if col in cat_col:
            fig,ax=plt.subplots(figsize=(6,5))
            sns.countplot(x=df[col],ax=ax)
            st.pyplot(fig)
            st.subheader(data['univariate']['cat_col'][col])
        else:
            #if pl=='kdeplot':
            fig,ax=plt.subplots(figsize=(6,5))
            sns.histplot(x=df[col],ax=ax,kde=True,color='red')
            st.pyplot(fig)
            st.subheader(data['univariate']['num_col']['kdeplot'][col])
    def bivariate(col):
        if col in cat_col:
            fig,ax=plt.subplots(figsize=(6,5))
            ct = pd.crosstab(df[col], df['returned'], normalize='index')*100
            ct.plot(kind='bar', stacked=True,colormap='RdBu_r',ax=ax,grid=True)
            plt.ylabel('Percentage %')
            st.pyplot(fig)
            st.subheader(data['bivariate']['cat_targ'][col+' vs returned'])
        else:
            fig,ax=plt.subplots(figsize=(6,5))
            sns.boxplot(x=cat_col[-1], y=col,data=df)
            st.pyplot(fig)
            st.subheader(data['bivariate']['num_targ'][col+' vs returned'])
    def multivariate(col):
        if mul_dict[col]%2!=0:
            fig=sns.pairplot(df[[i.strip() for i in col.split('vs')]],hue='returned',diag_kind='kde',corner=True)
            st.pyplot(fig)
            st.subheader(data['multivariate'][data_type][col])
        else:
            cols=[i.strip() for i in col.split('vs')][:-1]
            fig=sns.catplot(x=cols[0],hue='returned',col=cols[-1],kind='count',data=df)
            st.pyplot(fig)
            st.subheader(data['multivariate'][data_type][col])
    if type=='Univariate Analysis':
        data_type=st.selectbox('pick data type of columns',['numerical','categorical'])
        st.subheader(data_type)
        if data_type=='numerical':
            num=st.selectbox('pick a column',num_col)
            univariate(num)
        else:
            cat=st.selectbox('pick a column',cat_col)
            univariate(cat)
    elif type=='Bivariate Analysis':
        data_type=st.selectbox('pick data type of columns',['numerical vs target','categorical vs target'])
        st.subheader(data_type)
        if data_type=='numerical vs target':
            num=st.selectbox('pick a column',num_col2)
            bivariate(num)
        else:
            cat=st.selectbox('pick a column',cat_col[:-1])
            bivariate(cat)
    elif type=='Multivariate Analysis':
        data_type=st.selectbox('pick one',['all_numerical_columns','only_limited_columns'])
        st.subheader(data_type)
        if data_type=='all_numerical_columns':
            fig,ax=plt.subplots(figsize=(15,15))
            sns.heatmap(data=df[num_col].corr(),annot=True,vmin=-1,vmax=1,linewidths=1)
            st.pyplot(fig)
            st.subheader(data['multivariate'][data_type])
        else:
            mul_dict={'past_purchase_count vs past_return_rate vs returned':1,'used_coupon vs payment_method vs returned':2,'product_price vs discount_percent vs returned':3,'device_type vs shipping_method vs returned':4,'session_length_minutes vs num_product_views vs returned':5,'product_category vs shipping_method vs returned':6}
            l1=mul_dict.keys()
            mul_list=st.selectbox('choose one',l1)
            st.subheader(mul_list)
            multivariate(mul_list)