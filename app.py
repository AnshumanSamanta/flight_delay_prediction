import streamlit as st
import joblib
import pandas as pd
from datetime import datetime, timedelta

# Load models and encoders
xgb_reg = joblib.load('xgboost_model_complete.joblib')
log_reg = joblib.load('logistic_regression_model_complete.joblib')
scaler = joblib.load('scalar_complete.joblib')
label_encoders = joblib.load('label_encoder_complete.joblib')  # dict of encoders

# Sample airport and airline lists
airports = ['ABE', 'ABI', 'ABQ', 'ABR', 'ABY', 'ACK', 'ACT', 'ACV', 'ACY', 'ADK', 'ADQ', 'AEX', 'AGS', 'AKN', 'ALB', 'ALO', 'AMA', 'ANC', 'APN', 'ASE', 'ATL', 'ATW', 'AUS', 'AVL', 'AVP', 'AZO', 'BDL', 'BET', 'BFL', 'BGM', 'BGR', 'BHM', 'BIL', 'BIS', 'BJI', 'BLI', 'BMI', 'BNA', 'BOI', 'BOS', 'BPT', 'BQK', 'BQN', 'BRD', 'BRO', 'BRW', 'BTM', 'BTR', 'BTV', 'BUF', 'BUR', 'BWI', 'BZN', 'CAE', 'CAK', 'CDC', 'CDV', 'CEC', 'CHA', 'CHO', 'CHS', 'CID', 'CIU', 'CLD', 'CLE', 'CLL', 'CLT', 'CMH', 'CMI', 'CMX', 'CNY', 'COD', 'COS', 'COU', 'CPR', 'CRP', 'CRW', 'CSG', 'CVG', 'CWA', 'DAB', 'DAL', 'DAY', 'DBQ', 'DCA', 'DEN', 'DFW', 'DHN', 'DIK', 'DLG', 'DLH', 'DRO', 'DSM', 'DTW', 'DVL', 'EAU', 'ECP', 'EGE', 'EKO', 'ELM', 'ELP', 'ERI', 'ESC', 'EUG', 'EVV', 'EWN', 'EWR', 'EYW', 'FAI', 'FAR', 'FAT', 'FAY', 'FCA', 'FLG', 'FLL', 'FNT', 'FSD', 'FSM', 'FWA', 'GCC', 'GCK', 'GEG', 'GFK', 'GGG', 'GJT', 'GNV', 'GPT', 'GRB', 'GRI', 'GRK', 'GRR', 'GSO', 'GSP', 'GST', 'GTF', 'GTR', 'GUC', 'GUM', 'HDN', 'HIB', 'HLN', 'HNL', 'HOB', 'HOU', 'HPN', 'HRL', 'HSV', 'HYA', 'HYS', 'IAD', 'IAG', 'IAH', 'ICT', 'IDA', 'ILG', 'ILM', 'IMT', 'IND', 'INL', 'ISN', 'ISP', 'ITH', 'ITO', 'JAC', 'JAN', 'JAX', 'JFK', 'JLN', 'JMS', 'JNU', 'KOA', 'KTN', 'LAN', 'LAR', 'LAS', 'LAW', 'LAX', 'LBB', 'LBE', 'LCH', 'LEX', 'LFT', 'LGA', 'LGB', 'LIH', 'LIT', 'LNK', 'LRD', 'LSE', 'LWS', 'MAF', 'MBS', 'MCI', 'MCO', 'MDT', 'MDW', 'MEI', 'MEM', 'MFE', 'MFR', 'MGM', 'MHK', 'MHT', 'MIA', 'MKE', 'MKG', 'MLB', 'MLI', 'MLU', 'MMH', 'MOB', 'MOT', 'MQT', 'MRY', 'MSN', 'MSO', 'MSP', 'MSY', 'MTJ', 'MVY', 'MYR', 'OAJ', 'OAK', 'OGG', 'OKC', 'OMA', 'OME', 'ONT', 'ORD', 'ORF', 'ORH', 'OTH', 'OTZ', 'PAH', 'PBG', 'PBI', 'PDX', 'PHF', 'PHL', 'PHX', 'PIA', 'PIB', 'PIH', 'PIT', 'PLN', 'PNS', 'PPG', 'PSC', 'PSE', 'PSG', 'PSP', 'PUB', 'PVD', 'PWM', 'RAP', 'RDD', 'RDM', 'RDU', 'RHI', 'RIC', 'RKS', 'RNO', 'ROA', 'ROC', 'ROW', 'RST', 'RSW', 'SAF', 'SAN', 'SAT', 'SAV', 'SBA', 'SBN', 'SBP', 'SCC', 'SCE', 'SDF', 'SEA', 'SFO', 'SGF', 'SGU', 'SHV', 'SIT', 'SJC', 'SJT', 'SJU', 'SLC', 'SMF', 'SMX', 'SNA', 'SPI', 'SPS', 'SRQ', 'STC', 'STL', 'STT', 'STX', 'SUN', 'SUX', 'SWF', 'SYR', 'TLH', 'TOL', 'TPA', 'TRI', 'TTN', 'TUL', 'TUS', 'TVC', 'TWF', 'TXK', 'TYR', 'TYS', 'UST', 'VEL', 'VLD', 'VPS', 'WRG', 'WYS', 'XNA', 'YAK', 'YUM']
airlines = ['UA', 'AA', 'DL', 'WN', 'B6', 'AS', 'NK','WN','DL','EV','HA','MQ','VX']

st.set_page_config(page_title="Flight Delay Prediction", layout="wide")
st.title("🛫 Flight Delay Prediction App")

with st.form("delay_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        airline = st.selectbox("Airline", airlines)
        origin = st.selectbox("Origin Airport", airports)
        destination = st.selectbox("Destination Airport", airports)
        year = st.number_input("Year", 2000, 2030, 2024)

    with col2:
        month = st.selectbox("Month", list(range(1, 13)))
        day = st.selectbox("Day", list(range(1, 32)))
        week = st.number_input("Week Number", 1, 53, 20)
        wheels_off = st.time_input("Wheels Off Time", step=timedelta(minutes=1))

    with col3:
        sched_dep = st.time_input("Scheduled Departure Time (24hr)", step=timedelta(minutes=1))
        dep_time = st.time_input("Actual Departure Time (24hr)", step=timedelta(minutes=1))
        dep_delay = st.number_input("Departure Delay (mins)", -1440, 1440, 10)
        taxi_in = st.number_input("Taxi In (mins)", 0, 100, 5)
        taxi_out = st.number_input("Taxi Out (mins)", 0, 100, 10)

    submitted = st.form_submit_button("Predict")

    if submitted:
        # Convert times to integers like 0830
        def time_to_int(t):
            return int(t.strftime("%H%M"))

        wheels_off_int = time_to_int(wheels_off)
        sched_dep_int = time_to_int(sched_dep)
        dep_time_int = time_to_int(dep_time)

        # Input dictionary
        X_input = {
            'AIRLINE': airline,
            'ORIGIN_AIRPORT': origin,
            'DESTINATION_AIRPORT': destination,
            'DATE': datetime(year, month, day),
            'WEEK': week,
            'DEPARTURE_DELAY': dep_delay,
            'TAXI_IN': taxi_in,
            'TAXI_OUT': taxi_out,
            'WHEELS_OFF': wheels_off_int,
            'SCHEDULED_DEPARTURE': sched_dep_int,
            'DEPARTURE_TIME': dep_time_int,
        }

        # Encode categorical features
        for col in ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
            encoder = label_encoders[col]
            X_input[col] = encoder.transform([X_input[col]])[0]

        # Prepare model input
        X_model = pd.DataFrame([X_input])
        X_model.drop(columns=['DATE'], inplace=True)

        # Scale features
        X_scaled = scaler.transform(X_model)

        # Predictions
        is_delayed = log_reg.predict(X_scaled)[0]
        delay_time = int(xgb_reg.predict(X_scaled)[0])

        # Output
        st.markdown("### ✈️ Prediction Result")
        result_data = {
            "Delayed (Y/N)": ['Y' if is_delayed else 'N'],
            "Predicted Delay (mins)": [delay_time],
            "Entered Departure Delay (mins)": [dep_delay],
            "Day": [datetime(year, month, day).strftime('%a')],
            "Date": [datetime(year, month, day).strftime('%Y-%m-%d')],
            "Scheduled Departure": [sched_dep.strftime('%H:%M')],
            "Actual Departure": [dep_time.strftime('%H:%M')],
            "Wheels Off": [wheels_off.strftime('%H:%M')],
            "Origin": [origin],
            "Destination": [destination],
            "Taxi In (mins)": [taxi_in],
            "Taxi Out (mins)": [taxi_out],
        }
        st.table(pd.DataFrame(result_data))
