import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import sqlite3

# Kết nối đến cơ sở dữ liệu
conn = sqlite3.connect('history.db')
cursor = conn.cursor()

# Tạo bảng nếu chưa có
cursor.execute('''
CREATE TABLE IF NOT EXISTS run_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    LOAN NUMBER,
    MORTDUE NUMBER,
    VALUE  NUMBER,
    REASON  NUMBER,
    JOB  NUMBER,
    YOJ  NUMBER,
    DEROG  NUMBER,
    DELINQ  NUMBER,
    CLAGE  NUMBER,
    NINQ  NUMBER,
    CLNO  NUMBER,
    DEBTINC  NUMBER,
    SCORE  NUMBER,
    TIMESTAMP DATETIME DEFAULT CURRENT_TIMESTAMP
)
''')

def log_run(data):
    cursor.execute('''
    INSERT INTO run_history (LOAN, MORTDUE, VALUE, REASON, JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC, SCORE)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (float(data['LOAN'][0]), float(data['MORTDUE'][0]), float(data['VALUE'][0]),int( data['REASON'][0]), int(data['JOB'][0]), int(data['YOJ'][0]),float( data['DEROG'][0]),
          float(data['DELINQ'][0]), float(data['CLAGE'][0]), float(data['NINQ'][0]), float(data['CLNO'][0]), float(data['DEBTINC'][0]), data['SCORE'][0]))
    conn.commit()

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
model = pickle.load(open('G:\\APACHE_AIRFLOW\\dags\\RandomForest.pickle', 'rb'))
label_encode = pickle.load(open('G:\\APACHE_AIRFLOW\\dags\\LabelEncoder.pickle','rb'))

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
    
    REASON_VALUE = 0 if REASON_VAL == 'DebtCon' else 1
    JOB_VALUE = ['Office', 'ProfExe', 'Mgr', 'Self', 'Sales', 'Other'].index(JOB_VAL)
    
    # Tạo dataframe từ dữ liệu người dùng nhập vào
    data = {
        'LOAN': LOAN_VAL,
        'MORTDUE': MORTDUE_VAL,
        'VALUE': VALUE_VAL,
        'REASON': REASON_VALUE,
        'JOB': JOB_VALUE,
        'YOJ': YOJ_VAL,
        'DEROG': DEROG_VAL,
        'DELINQ': DELINQ_VAL,
        'CLAGE': CLAGE_VAL,
        'NINQ': NINQ_VAL,
        'CLNO': CLNO_VAL,
        'DEBTINC': DEBTINC_VAL
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
        output.loc[:, :] = scaler.transform(output)  # Chuẩn hóa dữ liệu
        # Dự đoán điểm tín dụng
        credit_score = compute_credit_score(model.predict_proba(output)[0][1])
        df['SCORE'] = credit_score
        log_run(df)

        # Hiển thị kết quả dự đoán
        if 500 <= credit_score <= 600:
            st.balloons()
            st.markdown('Your credit score is **GOOD**! Congratulations!')
            st.markdown('This credit score indicates that this person is likely to repay a loan, so the risk of giving them credit is low.')
        elif 300 <= credit_score < 500:
            st.markdown('Your credit score is **REGULAR**.')
            st.markdown('This credit score indicates that this person is likely to repay a loan, but can occasionally miss some payments. Meaning that the risk of giving them credit is medium.')
        elif credit_score < 300:
            st.markdown('Your credit score is **POOR**.')
            st.markdown('This credit score indicates that this person is unlikely to repay a loan, so the risk of lending them credit is high.')
        
        st.markdown('Your credit score is: ' + str(credit_score))

# Đóng kết nối
conn.close()
