import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Vibration Fault Diagnosis", layout="wide")

# ---------------------------------------------------------
# LOAD RESOURCES
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    # 1. Load Model
    with open('vibration_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # 2. Load Training History (For Line Graph)
    try:
        with open('training_history.pkl', 'rb') as f:
            history = pickle.load(f)
    except:
        history = None
        st.warning("training_history.pkl not found. Graphs may not display.")

    # 3. Load Evaluation Data (For Confusion Matrix)
    try:
        with open('model_evaluation.pkl', 'rb') as f:
            eval_data = pickle.load(f)
    except:
        eval_data = None
        st.warning("model_evaluation.pkl not found. Graphs may not display.")

    return model, history, eval_data

model, history, eval_data = load_resources()

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def extract_features(signal):
    features = []
    features.extend([
        np.mean(signal), np.std(signal), np.max(signal), np.min(signal),
        np.sqrt(np.mean(signal**2)), stats.skew(signal), stats.kurtosis(signal),
        np.mean(np.abs(signal)), np.sum(signal**2),
    ])
    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal))
    if len(fft_vals) > 2:
        dominant_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
        features.extend([freqs[dominant_idx], np.sum(fft_vals), np.std(fft_vals)])
    else:
        features.extend([0, 0, 0])
    return np.array(features)

label_map = {
    0: "Healthy", 1: "Planet crack (S1)", 2: "Ring Broken (S1)", 3: "Sun Chipped (S1)",
    4: "Planet 75% (S2)", 5: "Ring Missing (S2)", 6: "Sun Chipped 2nd (S2)",
    7: "Planet 2 Broken (S3)", 8: "Ring 2 Tooth (S3)", 9: "Sun 2 Taper (S3)"
}
target_names_full = [
    'Healthy', 'Planet crack (S1)', 'Ring Broken (S1)', 'Sun Chipped (S1)',
    'Planet 75% (S2)', 'Ring Missing (S2)', 'Sun Chipped 2nd (S2)',
    'Planet 2 Broken (S3)', 'Ring 2 Tooth (S3)', 'Sun 2 Taper (S3)'
]

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Navigation")

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Predict Fault", "📊 Model Analysis"])

# ==========================================
# TAB 1: FAULT PREDICTION
# ==========================================
with tab1:
    st.title("Gearbox Vibration Fault Diagnosis")
    st.markdown("Upload a CSV vibration signal file to detect faults.")

    window_size = st.sidebar.slider("Analysis Window Size", 100, 5000, 1000)
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=2)
            if df.shape[1] < 2:
                st.error("CSV must have at least 2 columns.")
            else:
                signal = df.iloc[:, 1].values
                if len(signal) >= window_size:
                    signal_window = signal[:window_size]
                else:
                    signal_window = signal
                
                features = extract_features(signal_window)
                prediction = model.predict([features])[0]
                probability = model.predict_proba([features])[0]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Class", label_map.get(prediction, "Unknown"))
                with col2:
                    health_index = probability[0] * 100
                    st.metric("Health Index", f"{health_index:.2f}%")

                st.write("---")
                st.subheader("Prediction Probability")
                prob_df = pd.DataFrame({'Fault Type': [label_map[i] for i in range(10)], 'Probability (%)': probability * 100})
                st.bar_chart(prob_df.set_index('Fault Type'))
                st.subheader("Signal Visualization")
                st.line_chart(signal_window)

        except Exception as e:
            st.error(f"Error: {e}")

# ==========================================
# TAB 2: MODEL ANALYSIS (GRAPHS)
# ==========================================
with tab2:
    st.title("Model Performance Analysis")
    st.markdown("Visualizing training history and evaluation metrics.")

    if history is None or eval_data is None:
        st.error("Required data files (training_history.pkl or model_evaluation.pkl) are missing from the repository.")
    else:
        col_a, col_b = st.columns(2)

        # --- GRAPH 1: TRAINING VS TESTING ACCURACY ---
        with col_a:
            st.subheader("Learning Curve")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(history['iterations'], history['train_acc'], 'b-', label='Training Accuracy')
            ax.plot(history['iterations'], history['test_acc'], 'r-', label='Test Accuracy')
            ax.set_title('Random Forest: Training vs Test Accuracy')
            ax.set_xlabel('Iterations (Number of Trees)')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

        # --- GRAPH 2: CONFUSION MATRIX ---
        with col_b:
            st.subheader("Confusion Matrix")
            cm = eval_data['confusion_matrix']
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names_full, yticklabels=target_names_full,
                        ax=ax, annot_kws={"size": 8})
            ax.set_title('Confusion Matrix (Random Forest)')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(fig)

        # --- ACCURACY TEXT SUMMARY ---
        st.info(f"Final Training Accuracy: {history['train_acc'][-1]*100:.2f}% | Final Test Accuracy: {history['test_acc'][-1]*100:.2f}%")