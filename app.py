import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from scipy.signal import resample

# ---------------------------------------------------------
# CONFIGURATION & LOAD MODEL
# ---------------------------------------------------------
st.set_page_config(page_title="Vibration Fault Diagnosis", layout="wide")

@st.cache_resource
def load_model():
    with open('vibration_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# ---------------------------------------------------------
# COPY YOUR FEATURE EXTRACTION LOGIC HERE
# ---------------------------------------------------------
def extract_features(signal):
    """Extract statistical features from signal (Exact same logic as training)"""
    features = []
    # Time domain features
    features.extend([
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.sqrt(np.mean(signal**2)),  # RMS
        stats.skew(signal),
        stats.kurtosis(signal),
        np.mean(np.abs(signal)),      # Mean absolute value
        np.sum(signal**2),            # Energy
    ])

    # Frequency domain features
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal))
    
    # Avoid divide by zero if signal is empty
    if len(fft_vals) > 2:
        dominant_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
        features.append(freqs[dominant_idx])
        features.append(np.sum(fft_vals))
        features.append(np.std(fft_vals))
    else:
        # Default values if signal is too short
        features.extend([0, 0, 0]) 

    return np.array(features)

# ---------------------------------------------------------
# LABEL MAPPING (Based on your code)
# ---------------------------------------------------------
label_map = {
    0: "Healthy",
    1: "Planet surface crack (S1)",
    2: "Ring Broken tooth (S1)",
    3: "Sun Chipped tooth (S1)",
    4: "Planet gear defect 75% (S2)",
    5: "Ring gear one tooth Missing (S2)",
    6: "Sun Gear Defect Chipped Tooth 2nd (S2)",
    7: "Planet 2 Broken tooth 180 (S3)",
    8: "RING 2 TOOTH 120 (S3)",
    9: "Sun Gear 2 tooth taper crack (S3)"
}

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("🔧 Gearbox Vibration Fault Diagnosis")
st.markdown("Upload a CSV vibration signal file to detect faults.")

st.sidebar.header("Settings")
window_size = st.sidebar.slider("Analysis Window Size", 100, 5000, 1000)

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # 1. Read CSV (Match your training logic: header=2, column 1)
        df = pd.read_csv(uploaded_file, header=2)
        
        if df.shape[1] < 2:
            st.error("CSV file must have at least 2 columns.")
        else:
            signal = df.iloc[:, 1].values
            
            # 2. Process Signal
            # We take the first 'window_size' points for prediction (similar to your training window)
            if len(signal) >= window_size:
                signal_window = signal[:window_size]
            else:
                st.warning(f"Signal is shorter than window size ({len(signal)} < {window_size}). Using available data.")
                signal_window = signal
            
            # 3. Extract Features
            features = extract_features(signal_window)
            
            # 4. Predict
            prediction = model.predict([features])[0]
            probability = model.predict_proba([features])[0]
            
            # 5. Display Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Class", label_map.get(prediction, "Unknown"))
                
            with col2:
                # Calculate Health Index (Probability of being Healthy - Class 0)
                health_index = probability[0] * 100
                st.metric("Health Index", f"{health_index:.2f}%")

            st.write("---")
            st.subheader("Detailed Probabilities")
            
            # Create a bar chart of probabilities
            prob_df = pd.DataFrame({
                'Fault Type': [label_map[i] for i in range(10)],
                'Probability (%)': probability * 100
            })
            
            st.bar_chart(prob_df.set_index('Fault Type'))

            # Plot the uploaded signal
            st.subheader("Signal Visualization")
            st.line_chart(signal_window)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Waiting for file upload...")

