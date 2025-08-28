import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from api.prediction import get_pred, mapping_species

st.set_page_config(page_title="Iris Prediction", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []
st.title("BTJ Academy - Iris Prediction App ")
st.write("Masukkan fitur iris lalu kirim ke FastAPI untuk memprediksi kelas")
st.divider()

tab_pred, tab_history = st.tabs(["Prediction", "History"])

with tab_pred:
    with st.form("predict_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.number_input("Sepal Length",  value=0.0, step=0.1)
            sepal_width  = st.number_input("Sepal Width",   value=0.0, step=0.1)
        with col2:
            petal_length = st.number_input("Petal Length",  value=0.0, step=0.1)
            petal_width  = st.number_input("Petal Width",   value=0.0, step=0.1)
        submit = st.form_submit_button("Predict")

    if submit:
        data = {
            "sepal_length": sepal_length,
            "sepal_width":  sepal_width,
            "petal_length": petal_length,
            "petal_width":  petal_width,
        }
        with st.spinner("Predicting..."):
            msg, result, species, probabilities = get_pred(data)
        if result:
            y = result[0] if len(result) > 0 else None
            prob_y = list(probabilities.values())[y] if y is not None else None
            st.success(f"Prediksi kelas: {mapping_species(y)}") 
            st.write(f"Probabilitas:")
            st.json(probabilities)
            st.session_state.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
                "result": y,
                "species": species,
                "probabilities": prob_y
            })
            st.toast("Prediksi berhasil.")
        else:
            st.error(f"Prediksi gagal: {msg}")
            st.session_state.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
                "result": None,
                "species": "-",
                "probabilities": {}
            })
            st.toast("Prediksi gagal")

with tab_history:
    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download History (CSV)", data=csv, file_name="history.csv", mime="text/csv")
        if st.button(" Clear History"):
            st.session_state.history = []
            st.toast("History cleared.")
    else:
        st.info("Belum ada riwayat prediksi.")
