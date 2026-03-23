import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import resample

# --- CONFIGURATION ---
# Must match the training script exactly
WINDOW_SIZE = 1000
TARGET_NAMES = [
    'Healthy', 'Planet crack (S1)', 'Ring Broken (S1)', 'Sun Chipped (S1)',
    'Planet 75% (S2)', 'Ring Missing (S2)', 'Sun Chipped 2nd (S2)',
    'Planet 2 Broken (S3)', 'Ring 2 Tooth (S3)', 'Sun 2 Taper (S3)'
]

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_model():
    try:
        # Load the model and scaler saved from your training script
        model = joblib.load('rf_gearbox_model_small.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("""
        **Model files not found!** 
        
        Please ensure you have run the training script and that the following files 
        are in the same folder as this app:
        1. `rf_gearbox_model.pkl`
        2. `feature_scaler.pkl`
        """)
        return None, None

# --- FEATURE EXTRACTION (Must match Training Script) ---
def extract_features(signal):
    features = []
    # Time Domain
    features.extend([
        np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
        np.sqrt(np.mean(signal**2)), stats.skew(signal), stats.kurtosis(signal),
        np.mean(np.abs(signal)), np.sum(signal**2),
    ])
    # Frequency Domain
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal))
    # Avoid division by zero if signal is flat
    if np.sum(fft_vals[1:len(fft_vals)//2]) == 0:
        dominant_idx = 1
    else:
        dominant_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
    
    features.append(freqs[dominant_idx])
    features.append(np.sum(fft_vals))
    features.append(np.std(fft_vals))
    return np.array(features)

# --- MAIN APP ---
st.set_page_config(page_title="Gearbox Fault Detector", layout="wide")
st.title("🔧 Gearbox Fault Diagnosis System")

# Load Model
model, scaler = load_model()
if model is None:
    st.stop()

# File Upload
uploaded_file = st.file_uploader("Upload Vibration CSV File", type=["csv"])

if uploaded_file is not None:
    # 1. ROBUST CSV READING
    # Try to detect if the file has headers or is raw numeric data
    try:
        # Read first few lines to check format
        df_check = pd.read_csv(uploaded_file, nrows=5)
        
        # Check if first cell is numeric
        try:
            float(df_check.iloc[0, 0])
            # It's numeric -> Read without header
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None)
        except ValueError:
            # It has headers/text -> Read with header
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # 2. COLUMN SELECTION
    col_names = df.columns.tolist()
    # Default to column index 1 (usually amplitude in 2-column files)
    default_idx = 1 if len(col_names) > 1 else 0
    
    selected_col = st.selectbox("Select the Vibration/Amplitude Column", col_names, index=default_idx)
    
    signal_raw = df[selected_col].values

    # 3. DATA CLEANING & TYPE FIXING (Fixes the Z_FFT error)
    # Handle Complex Numbers (common if user selects FFT column)
    if np.iscomplexobj(signal_raw):
        st.warning(f"⚠️ Column '{selected_col}' contains Complex/FFT data. Taking Absolute Magnitude.")
        signal_raw = np.abs(signal_raw)

    # Force conversion to numeric (handles text/headers mixed in data)
    try:
        signal_raw = pd.to_numeric(signal_raw, errors='coerce')
    except Exception as e:
        st.error(f"❌ Could not convert column '{selected_col}' to numbers.")
        st.stop()

    # Remove NaNs and Infinities
    signal_clean = signal_raw[np.isfinite(signal_raw)]

    # Warning if user likely picked the wrong column (FFT vs Time)
    if "FFT" in str(selected_col) or "Freq" in str(selected_col) or "Spectrum" in str(selected_col):
        st.warning("🔔 **Note:** You selected a Frequency/FFT column. This model requires **Raw Time-Series Vibration**. Select the 'Time' or 'Amplitude' column for best results.")

    # 4. PRE-PROCESSING (Fixes the Healthy=Fault issue)
    if len(signal_clean) < WINDOW_SIZE:
        st.error(f"Signal too short! Need at least {WINDOW_SIZE} points, got {len(signal_clean)}")
    else:
        # A. Resample to exactly 1000 points to match Training Window
        # This aligns the FFT frequency bins with what the model saw during training
        signal_resampled = resample(signal_clean, WINDOW_SIZE)
        
        # B. Normalize Amplitude (Z-score)
        # This removes sensitivity to different sensor scales (Volts vs g-force)
        signal_normalized = (signal_resampled - np.mean(signal_resampled)) / (np.std(signal_resampled) + 1e-9)

        # 5. EXTRACT FEATURES & PREDICT
        features = extract_features(signal_normalized)
        features_scaled = scaler.transform([features]) # Use the LOADED scaler

        prediction_idx = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Diagnosis Result")
            result_name = TARGET_NAMES[prediction_idx]
            
            if prediction_idx == 0:
                st.success(f"## ✅ {result_name}")
            else:
                st.error(f"## ⚠️ DETECTED FAULT: {result_name}")

        with col2:
            st.subheader("Model Confidence")
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(TARGET_NAMES, probabilities * 100, color='skyblue')
            ax.set_xlabel("Probability (%)")
            ax.set_xlim(0, 100)
            bars[prediction_idx].set_color('red' if prediction_idx != 0 else 'green')
            st.pyplot(fig)
            plt.close(fig)

        # --- DEBUGGING / DETAILS ---
        with st.expander("🔍 View Analysis Details"):
            st.write("**Normalized Signal Window (Used for Prediction):**")
            fig2, ax2 = plt.subplots()
            ax2.plot(signal_normalized, color='blue', lw=1)
            ax2.set_title("Normalized Signal (Z-score)")
            ax2.set_xlabel("Sample")
            ax2.set_ylabel("Amplitude")
            st.pyplot(fig2)
            plt.close(fig2)
            
            st.write("**Extracted Feature Vector (Scaled):**")
            feat_names = ['Mean', 'Std', 'Max', 'Min', 'RMS', 'Skew', 'Kurtosis',
                          'MAD', 'Energy', 'Dom_Freq', 'Spec_Energy', 'Spec_Std']
            feat_df = pd.DataFrame([features_scaled[0]], columns=feat_names)
            st.dataframe(feat_df.T.rename(columns={0: 'Value'}))
