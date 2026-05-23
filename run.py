from module  import *
from pred import *

org_df=pd.read_csv(r'sample_from_return.csv')
df=pd.read_csv(r'sample_from_return.csv')
df['product_price']=df['product_price'].abs()
df['session_length_minutes']=df['session_length_minutes'].abs()
df['num_product_views']=df['num_product_views'].abs()
df=df[df['discount_percent']>0]
df=df[df['past_return_rate']>0]
df=df[df['num_product_views']>0]
df.drop(columns='order_id',axis=1,inplace=True)
page = st.sidebar.radio('Go to', ['Home', 'EDA', 'Make Predictions'])

if page=='Home':
    add_bg_image("background_images/back_home.jpg")
    st.markdown("""
    <h1 style='text-align: center;'>E-Commerce Product Return Prediction</h1>
    <h3 style='text-align: center;'>Who Will Send It Back</h3>
    """, unsafe_allow_html=True)
    st.sidebar.title('navigation')
    st.subheader('Problem Statement')
    st.write('In Ecommerce Sectors product returns increase operational costs, affect inventory management, and reduce profitability. So identifying customers who are likely to return products helps businesses take preventive actions.')
    st.subheader('Objective')
    st.write('The main objective of this project is to build a machine learning model that predicts the likelihood of a customer returning a purchased product in an e-commerce setting, by analyzing customer behavioral patterns and purchase history using XGBoost — enabling businesses to take data-driven preventive actions and reduce return-related losses.')
elif page=='EDA':
    add_bg_image('background_images/eda.jpg')
    org=st.radio('',['original','manipulated'])
    if org=='original':
        fun(org_df)
    else:
        fun(df)
        analysis(df)
else:
    add_bg_image('background_images/pred.webp')
    pred(df)

