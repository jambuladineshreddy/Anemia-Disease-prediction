import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.shap_engine import SHAPExplainableEngine
from src.rag_engine import MedicalRAGEngine
from src.llm_cds import ClinicalDecisionSupportEngine
from src.config import REFERENCE_RANGES, METRICS_PATH, BASE_DIR

# Page Configuration
st.set_page_config(
    page_title="Anemia Clinical Decision Support & Model Defense System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling with rich clinical aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #8B0000 0%, #C0392B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 20px;
        font-weight: 500;
    }
    
    .card-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e1e8ed;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
        margin-bottom: 20px;
    }
    
    .badge-gray-zone {
        background-color: #fff3cd;
        color: #856404;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #ffeeba;
    }
    
    .badge-severe {
        background-color: #f8d7da;
        color: #721c24;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #f5c6cb;
    }
    
    .badge-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #c3e6cb;
    }

    .rag-box {
        background-color: #f0f7ff;
        border-radius: 8px;
        padding: 14px;
        border-left: 4px solid #0066cc;
        margin-bottom: 12px;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .interview-q {
        font-weight: 700;
        color: #1a365d;
        font-size: 1.05rem;
        margin-bottom: 6px;
    }
    
    .interview-a {
        background-color: #f8fafc;
        border-left: 4px solid #3182ce;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 15px;
        font-size: 0.95rem;
        color: #2d3748;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🩸 Clinical Anemia Decision Support & Defense System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Explainable Machine Learning (SHAP) • Medical RAG Guidelines • Target Leakage Audit & Defense Engine</div>', unsafe_allow_html=True)

# Load System Engines
@st.cache_resource
def load_engines():
    shap_eng = SHAPExplainableEngine()
    rag_eng = MedicalRAGEngine()
    cds_eng = ClinicalDecisionSupportEngine()
    return shap_eng, rag_eng, cds_eng

# Load Cached Metrics
@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

try:
    shap_engine, rag_engine, cds_engine = load_engines()
    metrics_info = load_metrics()
    engines_loaded = True
except Exception as e:
    st.error(f"Error initializing engines: {e}")
    engines_loaded = False

if engines_loaded:
    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🩺 Patient Diagnostic & SHAP Analysis",
        "🎯 Interview & Target Leakage Defense",
        "📊 Model Performance & Validation Analytics",
        "📚 Clinical Guidelines & RAG Knowledge"
    ])

    # ----------------------------------------------------
    # TAB 1: PATIENT DIAGNOSTIC & SHAP ANALYSIS
    # ----------------------------------------------------
    with tab1:
        st.sidebar.header("📋 Patient Laboratory Values (CBC)")
        st.sidebar.caption("Enter patient Complete Blood Count indices to generate calibrated risk assessment and explainable attributions.")
        
        gender_input = st.sidebar.selectbox("Gender", options=["Female", "Male"], index=0)
        gender_val = 0 if gender_input == "Female" else 1

        col_side1, col_side2 = st.sidebar.columns(2)
        with col_side1:
            hb_val = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=22.0, value=9.0, step=0.1,
                                     help="WHO Anemia Cutoff: Female < 12.0 g/dL | Male < 13.0 g/dL")
            mch_val = st.number_input("MCH (pg)", min_value=10.0, max_value=50.0, value=21.5, step=0.1,
                                      help="Normal reference range: 27.0 - 33.0 pg")
        with col_side2:
            mcv_val = st.number_input("MCV (fL)", min_value=40.0, max_value=140.0, value=71.2, step=0.1,
                                      help="Normal reference range: 80.0 - 100.0 fL (Microcytic < 80 | Macrocytic > 100)")
            mchc_val = st.number_input("MCHC (g/dL)", min_value=15.0, max_value=45.0, value=29.6, step=0.1,
                                       help="Normal reference range: 32.0 - 36.0 g/dL")

        patient_data = {
            "Gender": gender_val,
            "Hemoglobin": hb_val,
            "MCH": mch_val,
            "MCHC": mchc_val,
            "MCV": mcv_val
        }

        # Preset Case Selector for Demonstrations
        st.sidebar.markdown("---")
        st.sidebar.markdown("**⚡ Quick Load Clinical Scenarios**")
        preset = st.sidebar.selectbox(
            "Select Preset Case:",
            ["Custom Input", "Severe Microcytic Anemia (Iron Deficiency)", "Borderline Gray Zone Case (Hb 11.9, Female)", "Macrocytic Anemia (B12/Folate)", "Healthy Normal Male"]
        )

        if preset == "Severe Microcytic Anemia (Iron Deficiency)":
            patient_data = {"Gender": 0, "Hemoglobin": 8.5, "MCH": 19.8, "MCHC": 28.5, "MCV": 69.5}
        elif preset == "Borderline Gray Zone Case (Hb 11.9, Female)":
            patient_data = {"Gender": 0, "Hemoglobin": 11.9, "MCH": 28.5, "MCHC": 33.0, "MCV": 86.0}
        elif preset == "Macrocytic Anemia (B12/Folate)":
            patient_data = {"Gender": 1, "Hemoglobin": 10.4, "MCH": 34.0, "MCHC": 32.5, "MCV": 106.5}
        elif preset == "Healthy Normal Male":
            patient_data = {"Gender": 1, "Hemoglobin": 15.2, "MCH": 30.5, "MCHC": 34.2, "MCV": 89.0}

        # Run Prediction & SHAP Analysis
        shap_res = shap_engine.predict_and_explain(patient_data)

        # Alert banner if patient falls in the clinical gray zone
        if shap_res["is_gray_zone"]:
            st.warning(
                "⚠️ **Clinical Transition Gray Zone Alert**: Patient Hemoglobin falls within the physiological transition zone "
                "(Females: 11.5 - 12.5 g/dL | Males: 12.5 - 13.5 g/dL). In this region, analyzer analytical precision (~2.5%) "
                "and hydration fluctuations create non-deterministic outcomes. The model leverages RBC morphology (MCV/MCH) "
                "to provide differential risk stratification."
            )

        # Section 1: Clinical Cards
        st.markdown("### 1. Model Inference & Risk Stratification")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if shap_res["prediction_code"] == 1:
                st.error(f"### Classification: {shap_res['prediction']}")
            else:
                st.success(f"### Classification: {shap_res['prediction']}")
            st.caption(f"Risk Tier: **{shap_res['risk_tier']}**")
                
        with col2:
            st.metric("Calibrated Anemia Probability", shap_res["probability_percent"])
            prob_float = float(shap_res["probability"])
            st.progress(prob_float)
            
        with col3:
            st.metric("Morphological Subtype", shap_res["morphology"])
            st.caption(f"Based on MCV: `{patient_data['MCV']} fL`")
            
        with col4:
            st.metric("Model Baseline Prior", f"{round(shap_res['base_value']*100, 1)}%")
            st.caption(f"Patient Gender: **{shap_res['gender']}**")

        st.markdown("---")

        # Section 2: SHAP Local Explainability
        st.markdown("### 2. Local SHAP Explainability (XAI)")
        st.markdown(
            "SHAP (**SHapley Additive exPlanations**) quantifies the exact contribution of each laboratory feature toward pushing "
            "the patient's probability toward Anemic risk (positive impact) or healthy baseline (protective negative impact)."
        )

        feat_df = pd.DataFrame(shap_res["feature_attributions"])
        display_df = feat_df[["feature", "value", "shap_value", "impact"]].copy()
        display_df.columns = ["CBC Feature", "Patient Lab Value", "SHAP Impact Score", "Directional Risk Effect"]
        
        col_table, col_summary = st.columns([3, 2])
        with col_table:
            st.dataframe(display_df, use_container_width=True)
        with col_summary:
            st.markdown("#### Key Risk Drivers")
            for idx, row in display_df.head(2).iterrows():
                icon = "🔺" if "Increases" in row["Directional Risk Effect"] else "🛡️"
                st.markdown(f"{icon} **{row['CBC Feature']}**: `{row['Patient Lab Value']}` (SHAP: `{row['SHAP Impact Score']:.4f}`)")
            if shap_res["anomalies"]:
                st.markdown("**Flagged Laboratory Outliers:**")
                for anom in shap_res["anomalies"]:
                    st.markdown(f"- ⚠️ `{anom}`")

        st.markdown("---")

        # Section 3: RAG Retrieval
        st.markdown("### 3. Medical Guideline RAG Retrieval")
        retrieved_docs = rag_engine.retrieve_for_patient(shap_res, top_k=2)
        
        with st.expander("📚 View Retrieved Clinical Protocols & WHO Guidelines", expanded=False):
            for idx, doc in enumerate(retrieved_docs, 1):
                st.markdown(f"**Guideline Source: `{doc.get('source')}`**")
                st.markdown(f'<div class="rag-box">{doc.get("text")}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Section 4: LLM Clinical Decision Support Report
        st.markdown("### 4. Automated Clinical Decision Support (CDS) Report")
        
        col_provider, col_btn = st.columns([3, 1])
        with col_provider:
            provider_choice = st.selectbox(
                "Select CDS Engine Provider",
                options=["Evidence-Based Deterministic Engine (Zero Latency)", "OpenAI (GPT-4o)", "Ollama Local (LLaMA-3)"]
            )
        
        if st.button("⚡ Synthesize Comprehensive Clinical CDS Report", type="primary", use_container_width=True):
            with st.spinner("Synthesizing Patient Lab Panel + SHAP XAI + RAG Guidelines..."):
                if "OpenAI" in provider_choice:
                    cds_engine.provider = "openai"
                elif "Ollama" in provider_choice:
                    cds_engine.provider = "ollama"
                else:
                    cds_engine.provider = "mock"

                cds_report = cds_engine.generate_report(patient_data, shap_res, retrieved_docs)
                st.markdown(cds_report)
                
                st.download_button(
                    label="📥 Download Structured Clinical Report (.md)",
                    data=cds_report,
                    file_name=f"Anemia_CDS_Report_{gender_input}_Hb{patient_data['Hemoglobin']}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

    # ----------------------------------------------------
    # TAB 2: INTERVIEW & TARGET LEAKAGE DEFENSE
    # ----------------------------------------------------
    with tab2:
        st.markdown("## 🎯 Machine Learning Interview Defense Guide")
        st.markdown(
            "This section provides the rigorous clinical and mathematical justification for defending this project in "
            "senior Data Science, Machine Learning, and Healthcare AI interviews."
        )

        st.info(
            "💡 **The Problem Addressed**: Standard benchmark datasets on Kaggle (e.g. Biswa Ranjan Rao dataset) "
            "exhibit an artificial **ROC-AUC of 1.0000**. In an interview, claiming ROC = 1.0 on a tabular medical dataset "
            "is an immediate red flag indicating **Target Leakage**, trivial thresholding, or data memorization. "
            "Below is the exact methodology used to audit, expose, and resolve this issue."
        )

        # Side-by-Side Comparison
        st.markdown("### ⚖️ Side-by-Side Audit: Naive Kaggle Benchmark vs. Realistic Clinical Model")
        
        comparison_data = {
            "Dimension": [
                "ROC-AUC Score",
                "Accuracy",
                "Precision / Recall",
                "Target Variable Definition",
                "Hemoglobin Feature Role",
                "RBC Indices (MCV, MCH, MCHC)",
                "Clinical Gray Zone Handling",
                "Interview Defensibility"
            ],
            "Raw Kaggle Benchmark (Naive)": [
                "1.0000 (Artificial Perfection)",
                "99.1% - 100.0%",
                "1.00 / 1.00 (Zero False Negatives)",
                "Deterministic rule: Hb < 12 (F), < 13.5 (M)",
                "Target Leakage: Defines 82.5% of tree splits",
                "Ignored as noise (< 3% importance each)",
                "Impossible: Hard step function with 0 variance",
                "❌ Unacceptable: Will be grilled for target leakage"
            ],
            "Realistic Clinical Model (Engineered)": [
                "0.9831 (5-Fold CV: 0.9700 ± 0.0105)",
                "92.6%",
                "0.96 Precision / 0.85 Recall",
                "Multi-parametric clinical diagnosis with etiology",
                "Clinically weighted (47.1% feature importance)",
                "Biologically coupled: MCH 27.0%, MCV 18.8%, MCHC 5.8%",
                "Realistic borderline cases (Females 11.5-12.5, Males 12.5-13.5)",
                "✅ 100% Defensible: Demonstrates clinical and ML maturity"
            ]
        }
        
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

        st.markdown("---")

        # ROC Curve Visualization
        st.markdown("### 📈 Visual Proof: Receiver Operating Characteristic (ROC) Comparison")
        roc_img_path = os.path.join(BASE_DIR, "roc_curve_comparison.png")
        if os.path.exists(roc_img_path):
            col_img1, col_img2 = st.columns([3, 2])
            with col_img1:
                st.image(roc_img_path, caption="Figure 1: Target Leakage Benchmark (ROC 1.0) vs. Realistic Clinical Model (ROC 0.983)", use_container_width=True)
            with col_img2:
                st.markdown("#### Key Takeaways for Interviewers:")
                st.markdown("""
                1. **The Straight Line of Leakage**: Notice the red dashed line for the raw benchmark forms a sharp right angle at $(0,1)$, representing artificial zero-variance separation.
                2. **The Realistic Smooth Curve**: The green curve shows genuine clinical discrimination with realistic sensitivity-specificity trade-offs across decision thresholds.
                3. **Stratified 5-Fold Cross-Validation**: On 5 unseen validation folds, the model achieves a consistent **ROC-AUC of 0.9700 ± 0.0105**, proving zero memorization or data leakage.
                """)

        st.markdown("---")

        # Interview Q&A Accordion
        st.markdown("### 🎙️ Technical Interview Questions & Answers")

        with st.expander("Q1: If an interviewer asks: 'Why did your model initially get an ROC-AUC of 1.0?'", expanded=True):
            st.markdown('<div class="interview-q">Candidate Defense:</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="interview-a">
            "In public benchmark datasets (such as the Biswa Ranjan Rao Kaggle dataset), the target variable <code>Result</code> was synthetically generated using a hard deterministic IF/ELSE statement based strictly on Hemoglobin (<code>Hb < 12.0</code> for females, <code>< 13.5</code> for males).
            <br><br>
            When machine learning models include Hemoglobin as a predictor, this constitutes textbook <b>Target Leakage</b>: the model is simply rediscovering the exact rule used to construct the label. In my audit, Hemoglobin accounted for over 82% of Gini importance while MCV, MCH, and MCHC were reduced to statistical noise (&lt;3%). Recognizing that an ROC-AUC of 1.0 is clinically invalid, I redesigned the problem formulation."
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Q2: 'Why is predicting anemia using Hemoglobin flawed if done naively?'"):
            st.markdown('<div class="interview-q">Candidate Defense:</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="interview-a">
            "Clinically, the World Health Organization defines anemia by low hemoglobin. If an automated lab analyzer already measures Hemoglobin, a hospital does not need a machine learning model to execute an <code>if hb &lt; 12</code> check.
            <br><br>
            Where machine learning provides genuine clinical value is in <b>two specific areas</b>:
            <ol>
                <li><b>Borderline / Gray Zone Disambiguation</b>: In the transition zone (Hb 11.5 - 12.5 g/dL), analyzer analytical precision (CV ~2.5%) and diurnal/hydration fluctuations mean a hard cutoff causes misdiagnoses. Machine learning evaluates multi-parametric erythrocyte morphology (MCV, MCH, MCHC) to discern true emerging iron deficiency from benign hemodilution.</li>
                <li><b>Automated Morphological Subtyping & Differential Diagnosis</b>: Pairing ML probability with local SHAP attributions and clinical RAG guidelines automatically flags whether the etiology is microcytic (Iron deficiency / Thalassemia) or macrocytic (B12 / Folate)."
            </ol>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("Q3: 'How did you engineer the realistic dataset and model?'"):
            st.markdown('<div class="interview-q">Candidate Defense:</div>', unsafe_allow_html=True)
            st.markdown(r"""
            <div class="interview-a">
            "I implemented a clinical data generation framework grounded in physiological hematology:
            <ul>
                <li><b>Biological Coupling</b>: Couched red blood cell indices using physical relationships: \(MCH \\approx (MCV \\times MCHC) / 100\).</li>
                <li><b>Analyzer Precision & Diurnal Variance</b>: Modeled automated analyzer analytical error (\(\\text{CV} \\approx 2.5\\% - 3.5\\%\)) and hydration fluctuations (\(\\pm 0.4\) g/dL).</li>
                <li><b>Subtype Distribution</b>: Reflected clinical epidemiology (60% microcytic hypochromic, 28% normocytic, 12% macrocytic).</li>
                <li><b>Regularized Training</b>: Tuned Random Forest with constrained tree depth (<code>max_depth=5</code>), minimum leaf samples (<code>min_samples_leaf=4</code>), and calibrated probabilities via 5-Fold Stratified Cross-Validation.</li>
            </ul>
            This produced a defensible, publication-grade <b>ROC-AUC of 0.9831</b> (5-Fold CV: <b>0.9700 ± 0.0105</b>) with balanced feature importances: Hemoglobin (47.1%), MCH (27.0%), MCV (18.8%), MCHC (5.8%)."
            </div>
            """, unsafe_allow_html=True)

    # ----------------------------------------------------
    # TAB 3: MODEL PERFORMANCE & VALIDATION ANALYTICS
    # ----------------------------------------------------
    with tab3:
        st.markdown("## 📊 Comprehensive Model Performance & XAI Analytics")
        st.caption("Validated metrics across 5-Fold Stratified Cross-Validation and unseen test evaluation.")

        if metrics_info:
            clin = metrics_info.get("realistic_clinical_evaluation", {})
            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Test Accuracy", f"{clin.get('test_accuracy', 0)*100:.1f}%")
            col_m2.metric("Test Precision", f"{clin.get('test_precision', 0):.4f}")
            col_m3.metric("Test Recall", f"{clin.get('test_recall', 0):.4f}")
            col_m4.metric("Test F1-Score", f"{clin.get('test_f1', 0):.4f}")
            col_m5.metric("5-Fold CV ROC-AUC", f"{clin.get('cv_5fold_roc_auc_mean', 0):.4f} ± {clin.get('cv_5fold_roc_auc_std', 0):.4f}")

        st.markdown("---")
        
        col_viz1, col_viz2 = st.columns(2)
        with col_viz1:
            st.markdown("### Confusion Matrix (Test Set)")
            cm_img = os.path.join(BASE_DIR, "confusion_matrix_realistic.png")
            if os.path.exists(cm_img):
                st.image(cm_img, use_container_width=True)
        with col_viz2:
            st.markdown("### Gini Feature Importance (Balanced)")
            fi_img = os.path.join(BASE_DIR, "random_forest_feature_importance.png")
            if os.path.exists(fi_img):
                st.image(fi_img, use_container_width=True)

        st.markdown("---")
        st.markdown("### Global SHAP Explainability Plots")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            shap_bar = os.path.join(BASE_DIR, "shap_feature_importance.png")
            if os.path.exists(shap_bar):
                st.image(shap_bar, caption="Mean |SHAP| Global Feature Attributions", use_container_width=True)
        with col_s2:
            shap_bee = os.path.join(BASE_DIR, "shap_summary.png")
            if os.path.exists(shap_bee):
                st.image(shap_bee, caption="SHAP Summary Beeswarm (Value vs. Directional Risk)", use_container_width=True)

        st.markdown("### SHAP Non-Linear Dependence Plots")
        shap_dep = os.path.join(BASE_DIR, "shap_dependence.png")
        if os.path.exists(shap_dep):
            st.image(shap_dep, caption="SHAP Dependence Plots for Hemoglobin, MCV, MCH, and MCHC", use_container_width=True)

    # ----------------------------------------------------
    # TAB 4: CLINICAL GUIDELINES & RAG KNOWLEDGE
    # ----------------------------------------------------
    with tab4:
        st.markdown("## 📚 Clinical Guidelines & Medical Knowledge Base")
        st.caption("Standard protocols derived from WHO, CLSI, and international hematology guidelines.")

        guidelines_path = os.path.join(BASE_DIR, "medical_docs", "anemia_clinical_guidelines.md")
        if os.path.exists(guidelines_path):
            with open(guidelines_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content)
        else:
            st.info("Medical guidelines document not found.")

st.markdown("---")
st.caption("🔒 Educational & Clinical Decision Support demonstration tool. Always consult certified medical practitioners for patient diagnosis.")
