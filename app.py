import streamlit as st
import numpy as np
import requests
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K 
tf.compat.v1.disable_eager_execution()
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Higgs Boson", layout="centered")
features=['der_mass_mmc', 'der_mass_transverse_met_lep', 'der_mass_vis',
       'pri_tau_pt', 'der_met_phi_centrality', 'der_pt_ratio_lep_tau','der_mass_jet_jet',
       'der_deltaeta_jet_jet', 'der_sum_pt', 
       'der_prodeta_jet_jet', 'der_lep_eta_centrality', 'pri_jet_num',
       'pri_jet_leading_eta', 'der_pt_h', 'pri_met_sumet', 'pri_jet_all_pt',
       'der_deltar_tau_lep', 'pri_met', 'pri_jet_leading_pt',
       'pri_jet_subleading_eta', 'pri_lep_eta', 'pri_jet_leading_phi',
       'pri_jet_subleading_pt', 'pri_tau_eta', 'pri_jet_subleading_phi']

st.markdown("<h1 style='text-align: center;'>Higgs Boson signal prediction </h1>", unsafe_allow_html=True)     
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")  
        der_mass_mmc=st.number_input('1.der_mass_mmc')
        der_mass_transverse_met_lep=st.number_input('2.der_mass_transverse_met_lep')
        der_mass_vis=st.number_input('3.der_mass_vis')
        pri_tau_pt=st.number_input('4.pri_tau_pt')
        der_met_phi_centrality=st.number_input('5.der_met_phi_centrality')
        der_pt_ratio_lep_tau=st.number_input('6.der_pt_ratio_lep_tau')
        
        der_mass_jet_jet=st.number_input('7.der_mass_jet_jet')
        der_deltaeta_jet_jet=st.number_input('8.der_deltaeta_jet_jet')
        der_sum_pt=st.number_input('9.der_sum_pt')
        der_prodeta_jet_jet=st.number_input('10.der_prodeta_jet_jet')
        pri_jet_leading_eta=st.number_input('11.pri_jet_leading_eta')
        der_pt_h=st.number_input('12.der_pt_h')
        der_lep_eta_centrality=st.number_input('13.der_lep_eta_centrality')

        pri_jet_num=st.number_input('14.pri_jet_num')
        der_deltar_tau_lep=st.number_input('15.der_deltar_tau_lep')
        pri_jet_all_pt=st.number_input('16.pri_jet_all_pt')

        pri_jet_leading_pt=st.number_input('17.pri_jet_leading_pt')
        pri_met=st.number_input('18.pri_met')
        pri_jet_subleading_eta=st.number_input('19.pri_jet_subleading_eta')
        pri_met_sumet=st.number_input('20.pri_met_sumet')
        pri_jet_leading_phi=st.number_input('21.pri_jet_leading_phi')
        pri_lep_eta=st.number_input('22.pri_lep_eta')
        pri_jet_subleading_pt=st.number_input('23.pri_jet_subleading_pt')
        pri_tau_eta=st.number_input('24.pri_tau_eta')
        
        submit = st.form_submit_button("Predict")
        if submit:
            model=load_model('Model/hbkfold_kerasModel.h5')
            data=np.array(features).reshape(1,-1)
            scaler=MinMaxScaler()
            X=scaler.fit_transform(data)
            pred=model.predict(X)
            if pred[0] == 0:
                result='Background'
            else:
                result='Signal'     
            st.write(f"The prediction is: {result}")    

if __name__ == '__main__':
    main()


