import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import resample

# --- CONFIGURATION (Must match Training Script) ---
WINDOW_SIZE = 1000  # CRITICAL: Must match training window size
TARGET_NAMES = [
    'Healthy', 'Planet crack (S1)', 'Ring Broken (S1)', 'Sun Chipped (S1)',
    'Planet 75% (S2)', 'Ring Missing (S2)', 'Sun Chipped 2nd (S2)',
    'Planet 2 Broken (S3)', 'Ring 2 Tooth (S3)', 'Sun 2 Taper (S3)'
]

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_model():
    try:
        
        model = joblib.load('rf_gearbox_model_small.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model files not found! Please run the training script first to generate 'rf_gearbox_model.pkl' and 'feature_scaler.pkl'")
        return None, None

# --- FEATURE EXTRACTION (Exact Copy from Training) ---
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
    dominant_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
    features.append(freqs[dominant_idx])
    features.append(np.sum(fft_vals))
    features.append(np.std(fft_vals))
    return np.array(features)

# --- MAIN APP ---
st.set_page_config(page_title="Gearbox Fault Detector", layout="wide")
st.title("🔧 Gearbox Fault Diagnosis System")

model, scaler = load_model()
if model is None:
    st.stop()

uploaded_file = st.file_uploader("Upload Vibration CSV File", type=["csv"])

if uploaded_file is not None:
    # 1. READ FILE ROBUSTLY
    # Try to detect header. If the first few rows aren't numeric, skip them.
    try:
        # Attempt reading without header first to inspect
        df_check = pd.read_csv(uploaded_file, nrows=5)
        # Check if first column looks like a float
        try:
            float(df_check.iloc[0, 0])
            # If it worked, read normally
            uploaded_file.seek(0) # Reset pointer
            df = pd.read_csv(uploaded_file, header=None)
        except ValueError:
            # If failed, assume header exists
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # 2. SELECT SIGNAL COLUMN
    # We assume user wants to analyze a specific column. Default to column 1 or 2 (index 1).
    col_names = df.columns.tolist()
    selected_col = st.selectbox("Select the Vibration/Amplitude Column", col_names, index=1 if len(col_names) > 1 else 0)
    
    signal_raw = df[selected_col].values

    # Remove non-numeric values just in case
    signal_clean = signal_raw[np.isfinite(signal_raw)]

    if len(signal_clean) < WINDOW_SIZE:
        st.error(f"File too short! Need at least {WINDOW_SIZE} data points, got {len(signal_clean)}")
    else:
        # 3. PRE-PROCESSING (THE FIX)
        
        # A. Resample to exactly 1000 points to match Training Window
        # This ensures FFT features are calculated on the same basis
        signal_resampled = resample(signal_clean, WINDOW_SIZE)
        
        # B. Normalize Amplitude (Z-score)
        # This fixes the issue where your sensor outputs different voltage/units than training data
        signal_normalized = (signal_resampled - np.mean(signal_resampled)) / (np.std(signal_resampled) + 1e-9)

        # 4. EXTRACT FEATURES
        features = extract_features(signal_normalized)
        features_scaled = scaler.transform([features]) # Apply the saved scaler!

        # 5. PREDICT
        prediction_idx = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]

        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Diagnosis Result")
            result_name = TARGET_NAMES[prediction_idx]
            
            # Color coding
            if prediction_idx == 0:
                st.success(f"## ✅ {result_name}")
            else:
                st.error(f"## ⚠️ DETECTED FAULT: {result_name}")

        with col2:
            st.subheader("Model Confidence")
            # Create a bar chart of probabilities
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(TARGET_NAMES, probabilities * 100, color='skyblue')
            ax.set_xlabel("Probability (%)")
            ax.set_xlim(0, 100)
            
            # Highlight the winner
            bars[prediction_idx].set_color('red' if prediction_idx != 0 else 'green')
            
            st.pyplot(fig)

        # --- DEBUGGING INFO (To help you understand why) ---
        with st.expander("🔍 Debug: Feature Analysis"):
            st.write("**Extracted Features (Normalized):**")
            feat_df = pd.DataFrame([features_scaled[0]], columns=[
                'Mean', 'Std', 'Max', 'Min', 'RMS', 'Skew', 'Kurtosis',
                'MAD', 'Energy', 'Dom_Freq', 'Spec_Energy', 'Spec_Std'
            ])
            st.dataframe(feat_df.T.rename(columns={0: 'Value'}))
            
            st.write("**Signal Visualization (Normalized):**")
            fig2, ax2 = plt.subplots()
            ax2.plot(signal_normalized, color='blue')
            ax2.set_title("Normalized Signal Window (Used for Prediction)")
            ax2.set_xlabel("Sample")
            ax2.set_ylabel("Amplitude (Z-score)")
            st.pyplot(fig2)
            
            st.info("💡 **Tip:** If 'Healthy' shows as a fault, check if your signal shape (skewness/kurtosis) is very different from the training data.")
