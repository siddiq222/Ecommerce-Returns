from module import *

with open(r'xgb_model.pkl','rb') as file1:
    model=pickle.load(file1)
    file1.close()
with open(r'preprocessing_models/onehotencoding.pkl','rb') as file2:
    ohe=pickle.load(file2)
    file2.close()
with open(r'preprocessing_models/ordinalencoding.pkl','rb') as file3:
    oe=pickle.load(file3)
    file3.close()
with open(r'preprocessing_models/yeojohnson.pkl','rb') as file4:
    trans=pickle.load(file4)
    file4.close()
with open(r'preprocessing_models/standardscaler.pkl','rb') as file5:
    scale=pickle.load(file5)
    file5.close()

def pred(df):
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
                background-color: rgba(14, 97, 78, 0.3) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
                border-radius: 6px !important;
            }

            /* +/- buttons */
            [data-testid="stNumberInput"] button {
                background-color: rgba(14, 97, 78, 0.3) !important;
                color: white !important;
                border: 1px solid rgba(255, 255, 255, 0.15) !important;
            }

            [data-testid="stNumberInput"] button:hover {
                background-color: rgba(14, 97, 78, 0.5) !important;
                box-shadow: none !important;
            }
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
    d={}
    age=st.slider('Customer Age',0,100)
    d['customer_age']=[age]
    price=st.number_input('product price at purchace')
    d['product_price']=[price]
    disc=st.slider('discount_percentage',0,100)
    d['discount_percent']=[disc]
    rat=st.slider('product rating',0,5)
    d['product_rating']=[rat]
    ppc=st.number_input('past purchase count')
    d['past_purchase_count']=[ppc]
    prr=st.slider('past return rate',0,100)
    d['past_return_rate']=[prr]
    ddd=st.number_input('delivery delay days',-10,10)
    d['delivery_delay_days']=[ddd]
    slm=st.number_input('session length in minutes')
    d['session_length_minutes']=[slm]
    npv=st.number_input('number of product views',0,)
    d['num_product_views']=[npv]
    dt=st.selectbox('device type',df['device_type'].unique())
    d['device_type']=[dt]
    pc=st.selectbox('product cateogry',df['product_category'].unique())
    d['product_category']=[pc]
    sm=st.selectbox('shipping method',df['shipping_method'].unique())
    d['shipping_method']=[sm]
    pm=st.selectbox('payment method',df['payment_method'].unique())
    d['payment_method']=[pm]
    uc=st.selectbox('used coupon',['yes','no'])
    if uc=='yes':
        d['used_coupon']=[1]
    else:
        d['used_coupon']=[0]


    ddff=pd.DataFrame(d)
    num_col=list(ddff.iloc[:,:9].columns)
    xyz2=pd.DataFrame(ohe.transform(ddff[['device_type','product_category','payment_method']]).toarray(),columns=ohe.get_feature_names_out())
    ddff['shipping_method']=oe.transform(ddff[['shipping_method']])
    ddff.drop(columns=['device_type','product_category','payment_method'],axis=1,inplace=True)
    ddff=pd.concat([ddff,xyz2],axis=1)
    ddff[num_col]=trans.transform(ddff[num_col])
    ddff[num_col]=scale.transform(ddff[num_col])

    if st.button('predict'):
        y_pred=model.predict(ddff)
        if y_pred==0:
            st.success('no return risk')
        else:
            st.warning('return risk')
