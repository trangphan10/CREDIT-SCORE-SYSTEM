import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Cấu hình trang Streamlit
st.set_page_config(
    page_title='Credit Score Application',
    page_icon='💰',
    layout='wide',
    initial_sidebar_state='auto',
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': '''
            The application is developed with the purpose of providing personal credit scores for loan requests at the bank.
        '''
    }
)

# Tải mô hình và bộ chuẩn hóa
scaler = pickle.load(open('G:\\APACHE_AIRFLOW\\dags\\StandardScaler.pickle', 'rb'))
model = pickle.load(open('G:\\APACHE_AIRFLOW\\dags\\LogisticRegression.pickle', 'rb'))
label_encode = pickle.load(open('G:\\APACHE_AIRFLOW\\dags\\LabelEncoder.pickle','rb'))

# Hàm biến đổi dữ liệu đầu vào
def transform(x):
    x.dropna(axis=0, how='any', inplace=True)
    x.drop('DEBTINC', axis=1, inplace=True)
    return x

def compute_credit_score(p):
    factor = 25 / np.log(2)
    offset = 600 - factor * np.log(50)
    val = (1 - p) / p
    score = offset + factor * np.log(val)
    return round(score, 2)

# Tiêu đề ứng dụng
st.title('Credit Score Analysis of Customers')

# Hàm nhập liệu từ người dùng
def user_input_data():
    LOAN_VAL = st.sidebar.slider('LOAN', 1, 90000, 1000, 100)
    MORTDUE_VAL = st.sidebar.slider('MORTDUE', 0, 400000, 2000, 100)
    VALUE_VAL = st.sidebar.slider('VALUE', 0, 855909, 8000, 100)
    REASON_VAL = st.sidebar.selectbox('REASON', ['DebtCon', 'HomeImp'])
    JOB_VAL = st.sidebar.selectbox('JOB', ['Office', 'ProfExe', 'Mgr', 'Self', 'Sales', 'Other'])
    YOJ_VAL = st.sidebar.slider('YOJ', 0, 45, 0, 1)
    DELINQ_VAL = st.sidebar.slider('DELINQ', 0, 45, 0, 1)
    DEROG_VAL = st.sidebar.slider('DEROG', 0, 10, 0, 1)
    CLAGE_VAL = st.sidebar.slider('CLAGE', 0, 1170, 0, 1)
    NINQ_VAL = st.sidebar.slider('NINQ', 0, 20, 0, 1)
    CLNO_VAL = st.sidebar.slider('CLNO', 0, 75, 0, 1)
    DEBTINC_VAL = st.sidebar.slider('DEBTINC', 0, 205, 0, 1)
    
    debtcon = 0
    homeimp = 0
    office = 0 
    profexe = 0 
    mgr = 0 
    self = 0 
    sales = 0 
    other = 0 
    if REASON_VAL == 'DebtCon': 
        debtcon = 1 
    else : 
        homeimp = 0 
        
    if JOB_VAL == 'Office': 
        office = 1 
    elif JOB_VAL == 'ProfExe': 
        profexe = 1 
    elif JOB_VAL == 'Mgr': 
        mgr = 1 
    elif JOB_VAL == 'Self': 
        self = 1 
    elif JOB_VAL == 'Sales': 
        sales = 1 
    elif JOB_VAL == 'Other': 
        other = 1
        
        
    # Tạo dataframe từ dữ liệu người dùng nhập vào
    data = {
        'LOAN': LOAN_VAL,
        'MORTDUE': MORTDUE_VAL,
        'VALUE': VALUE_VAL,
        'YOJ': YOJ_VAL,
        'DEROG': DEROG_VAL,
        'DELINQ': DELINQ_VAL,
        'CLAGE': CLAGE_VAL,
        'NINQ': NINQ_VAL,
        'CLNO': CLNO_VAL,
        'DEBTINC': DEBTINC_VAL,
        'REASON_DebtCon': debtcon,
        'REASON_HomeImp': homeimp,
        'JOB_Mgr':mgr, 
        'JOB_Office':office, 
        'JOB_Other':other,
        'JOB_ProfExe':profexe,
        'JOB_Sales':sales,
        'JOB_Self':self
    }
    input_data = pd.DataFrame(data, index=[0])
    
    return input_data

# Giao diện ứng dụng
col1, col2 = st.columns([4, 6])
df = user_input_data()

with col2:
    x1 = [0, 6, 0]
    x2 = [0, 4, 0]
    x3 = [0, 2, 0]
    y = ['0', '1', '2']

    f, ax = plt.subplots(figsize=(5, 2))

    p1 = sns.barplot(x=x1, y=y, color='#3EC300')
    p1.set(xticklabels=[], yticklabels=[])
    p1.tick_params(bottom=False, left=False)
    p2 = sns.barplot(x=x2, y=y, color='#FAA300')
    p2.set(xticklabels=[], yticklabels=[])
    p2.tick_params(bottom=False, left=False)
    p3 = sns.barplot(x=x3, y=y, color='#FF331F')
    p3.set(xticklabels=[], yticklabels=[])
    p3.tick_params(bottom=False, left=False)

    plt.text(0.7, 1.05, "POOR", horizontalalignment='left', size='medium', color='white', weight='semibold')
    plt.text(2.5, 1.05, "REGULAR", horizontalalignment='left', size='medium', color='white', weight='semibold')
    plt.text(4.7, 1.05, "GOOD", horizontalalignment='left', size='medium', color='white', weight='semibold')

    ax.set(xlim=(0, 6))
    sns.despine(left=True, bottom=True)

    st.pyplot(f)

with col1:
    if st.button('Make Prediction'):
        output = df.copy()
        output = transform(output)
        label = LabelEncoder()
        for column in output.columns:
            if output[column].dtype == 'object':
                output[column] = label_encode.fit_transform(output[column])
        scaled_features = scaler.fit_transform(output)

        scaled_features_df = pd.DataFrame(scaled_features, index=output.index, columns=output.columns)
        credit_score = compute_credit_score(model.predict_proba(output)[0][1])
        if credit_score < 300:
            st.balloons()
            t1 = plt.Polygon([[5, 0.5], [5.5, 0], [4.5, 0]], color='black')
            st.markdown('Your credit score is **GOOD**! Congratulations!')
            st.markdown('This credit score indicates that this person is likely to repay a loan, so the risk of giving them credit is low.')
        elif 300 <= credit_score < 500:
            t1 = plt.Polygon([[3, 0.5], [3.5, 0], [2.5, 0]], color='black')
            st.markdown('Your credit score is **REGULAR**.')
            st.markdown('This credit score indicates that this person is likely to repay a loan, but can occasionally miss some payments. Meaning that the risk of giving them credit is medium.')
        elif 500 <= credit_score <= 600:
            t1 = plt.Polygon([[1, 0.5], [1.5, 0], [0.5, 0]], color='black')
            st.markdown('Your credit score is **POOR**.')
            st.markdown('This credit score indicates that this person is unlikely to repay a loan, so the risk of lending them credit is high.')
        
        plt.gca().add_patch(t1)
        st.pyplot(f)
        
