import os
import sys
import json
import requests

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL,
    OLLAMA_BASE_URL, OLLAMA_MODEL
)


class ClinicalDecisionSupportEngine:
    def __init__(self, provider=LLM_PROVIDER):
        self.provider = provider

    def build_clinical_prompt(self, patient_dict, shap_result, retrieved_docs):
        """Constructs an augmented clinical prompt combining ML + SHAP + RAG context."""
        
        # 1. Format SHAP Drivers
        shap_summary = []
        for feat in shap_result.get("feature_attributions", []):
            direction = "Increased risk (+)" if feat["shap_value"] > 0 else "Decreased risk (-)"
            shap_summary.append(
                f"- **{feat['feature']}**: {feat['value']} (SHAP Impact: {feat['shap_value']:.4f} -> {direction})"
            )
        shap_text = "\n".join(shap_summary)
        
        # 2. Format RAG Medical Context
        rag_context = []
        for idx, doc in enumerate(retrieved_docs, 1):
            rag_context.append(f"--- Medical Guideline Excerpt #{idx} (Source: {doc.get('source', 'WHO Guide')}) ---\n{doc.get('text', '')}")
        rag_text = "\n\n".join(rag_context)

        gray_note = "PATIENT IS IN PHYSIOLOGICAL TRANSITION GRAY ZONE (borderline hemoglobin range)." if shap_result.get("is_gray_zone") else "Standard risk profile."

        prompt = f"""You are an expert Hematology Clinical Decision Support (CDS) AI system. 
Analyze the following patient lab panel, Machine Learning prediction, SHAP Explainable AI feature importance, and retrieved WHO/Clinical guidelines to generate a structured medical report.

### 1. PATIENT DEMOGRAPHICS & COMPLETE BLOOD COUNT (CBC):
- Gender: {shap_result.get('gender')}
- Hemoglobin (Hb): {patient_dict.get('Hemoglobin')} g/dL
- Mean Corpuscular Volume (MCV): {patient_dict.get('MCV')} fL
- Mean Corpuscular Hemoglobin (MCH): {patient_dict.get('MCH')} pg
- Mean Corpuscular Hemoglobin Concentration (MCHC): {patient_dict.get('MCHC')} g/dL

### 2. MACHINE LEARNING & SHAP EXPLAINABILITY ENGINE:
- Predicted Classification: {shap_result.get('prediction')}
- Calibrated Risk Probability: {shap_result.get('probability_percent')}
- Risk Tier: {shap_result.get('risk_tier')}
- Borderline Clinical Gray Zone: {gray_note}
- Calculated Morphological Subtype: {shap_result.get('morphology')}
- Reference Anomalies Flagged: {', '.join(shap_result.get('anomalies', []))}

#### Local SHAP Feature Contributions:
{shap_text}

### 3. RETRIEVED MEDICAL GUIDELINES (RAG CONTEXT):
{rag_text}

---
### INSTRUCTIONS FOR CLINICAL REPORT GENERATION:
Provide a clear, professional, structured report with the following sections:

1. **Executive Summary & Clinical Findings**: Synthesize the CBC values, ML model prediction, risk tier, and morphological classification ({shap_result.get('morphology')}). Address borderline status if applicable.
2. **SHAP Explainability & Risk Factor Analysis**: Explain WHY the model assigned a {shap_result.get('probability_percent')} risk score based on key feature attributions (how Hemoglobin, MCV, MCH, and MCHC interacted).
3. **Differential Diagnosis & Medical Evidence Alignment**: Reference retrieved clinical guidelines to outline likely differential diagnoses (e.g., Iron Deficiency Anemia vs. Thalassemia Trait vs. B12 Deficiency vs. Borderline Hemodilution).
4. **Recommended Follow-up Diagnostic Tests**: List essential confirmatory laboratory workups (e.g., Serum Ferritin, TIBC, Reticulocyte Count, Peripheral Smear).
5. **Medical Safety Disclaimer**: Explicitly state that this analysis is an automated decision support tool intended for clinician review and not a standalone medical diagnosis.
"""
        return prompt

    def generate_report(self, patient_dict, shap_result, retrieved_docs):
        """Generates clinical decision support report using configured LLM or fallback."""
        prompt = self.build_clinical_prompt(patient_dict, shap_result, retrieved_docs)
        
        # 1. Try OpenAI API if key available
        if self.provider == "openai" and OPENAI_API_KEY:
            try:
                import openai
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a clinical decision support system for hematology."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API call failed: {e}. Falling back to Rule-Based CDS Generator.")
                
        # 2. Try Ollama if configured
        elif self.provider == "ollama":
            try:
                res = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                    timeout=30
                )
                if res.status_code == 200:
                    return res.json().get("response", "")
            except Exception as e:
                print(f"Ollama call failed: {e}. Falling back to Rule-Based CDS Generator.")

        # 3. Intelligent Deterministic Clinical Decision Support Generator (Fallback)
        return self._generate_fallback_report(patient_dict, shap_result, retrieved_docs)

    def _generate_fallback_report(self, patient_dict, shap_result, retrieved_docs):
        """Generates a structured, evidence-backed clinical decision report when external LLM is offline."""
        prediction = shap_result.get("prediction")
        prob = shap_result.get("probability_percent")
        risk_tier = shap_result.get("risk_tier", "Standard Risk Profile")
        morphology = shap_result.get("morphology")
        gender = shap_result.get("gender")
        anomalies = shap_result.get("anomalies", [])
        is_gray = shap_result.get("is_gray_zone", False)
        
        top_driver = shap_result.get("feature_attributions", [{}])[0]
        driver_name = top_driver.get("feature", "Hemoglobin")
        driver_val = top_driver.get("value", "")

        gray_section = ""
        if is_gray:
            gray_section = """
> ⚠️ **Clinical Gray Zone Alert**: Patient hemoglobin falls within the physiological borderline transition zone. In this threshold region, day-to-day hydration shifts, analyzer CV (~2.5%), and baseline individual variation necessitate correlation with RBC morphology (MCV/MCH) rather than isolated threshold cutoff.
"""

        report = f"""## 🩺 Clinical Decision Support (CDS) Report

### 1. Executive Summary & Clinical Findings
- **Patient Profile**: {gender} adult
- **ML Classification**: **{prediction}** (Probability: **{prob}**)
- **Risk Stratification**: **{risk_tier}**
- **Morphological Subtyping**: **{morphology} Anemia**
- **Flagged Lab Anomalies**: {', '.join(anomalies) if anomalies else 'All indices within normal reference intervals'}
{gray_section}

### 2. SHAP Explainability & Risk Factor Analysis
The Random Forest classifier assigned a **{prob}** likelihood of anemia. The primary drivers derived from SHAP tree explainability include:
- **Primary Risk Driver**: `{driver_name}` = `{driver_val}` (SHAP Attribution: `{top_driver.get('shap_value', 0):.4f}`)
- **Pathology Interpretation**: The main feature pushing the model toward predicting **{prediction}** is the `{driver_name}` level. Combined with an MCV of `{patient_dict.get('MCV')} fL` and MCH of `{patient_dict.get('MCH')} pg`, the erythrocyte indices indicate a **{morphology}** cellular profile.

### 3. Differential Diagnosis & Medical Evidence Alignment
Based on the retrieved WHO Clinical Guidelines for **{morphology} Anemia**:
"""
        if morphology == "Microcytic Hypochromic":
            report += """
1. **Iron Deficiency Anemia (IDA)**: Primary diagnostic hypothesis given low MCV (<80 fL) and hypochromic indices.
2. **Thalassemia Trait (Alpha/Beta)**: Secondary differential if microcytosis is pronounced relative to mild anemia.
3. **Anemia of Chronic Disease (Early Stage)**: Requires evaluation of inflammatory state.
"""
        elif morphology == "Macrocytic":
            report += """
1. **Vitamin B12 / Folate Deficiency**: Leading cause of macrocytosis (MCV >100 fL).
2. **Megaloblastic Anemia**: Potential impaired DNA maturation.
3. **Hepatic Dysfunction / Hypothyroidism**: Secondary non-megaloblastic causes.
"""
        else:
            report += """
1. **Normocytic Anemia (Early Chronic Disease / Renal Impairment)**: Decreased erythropoietin production or chronic inflammation.
2. **Acute Blood Loss or Hemolysis**: Erythrocyte size remains normal (80-100 fL).
3. **Physiological Dilution**: In borderline cases, plasma volume expansion (e.g. overhydration or athletic conditioning) can lower hemoglobin without pathological anemia.
"""

        report += f"""
### 4. Recommended Follow-up Laboratory Diagnostics
To confirm the underlying etiology, the following workups are recommended:
"""
        if morphology == "Microcytic Hypochromic":
            report += """- [ ] **Iron Panel**: Serum Ferritin, Total Iron Binding Capacity (TIBC), Serum Iron, Transferrin Saturation.
- [ ] **Hemoglobin Electrophoresis**: To evaluate Thalassemia trait if ferritin is normal.
- [ ] **Peripheral Blood Smear**: To inspect microcytes, central pallor, and anisocytosis.
"""
        elif morphology == "Macrocytic":
            report += """- [ ] **Serum Vitamin B12 & RBC Folate Levels**
- [ ] **Peripheral Blood Smear**: Check for hypersegmented neutrophils.
- [ ] **Liver Function Tests (LFTs) & TSH**
"""
        else:
            report += """- [ ] **Reticulocyte Count**: Evaluate bone marrow erythropoietic response.
- [ ] **Renal Function Tests**: Serum Creatinine, Blood Urea Nitrogen (BUN).
- [ ] **Inflammatory Markers**: High-sensitivity CRP, ESR.
- [ ] **Repeat CBC in 2-4 Weeks**: To rule out transient physiological fluctuation.
"""

        report += """
---
> 🔒 **Medical Safety Disclaimer**: *This report was generated by an automated Explainable AI (XAI) & RAG Clinical Decision Support system. It is designed solely for educational and decision-support assistance for healthcare providers and does NOT constitute a binding medical diagnosis.*
"""
        return report


if __name__ == "__main__":
    from src.shap_engine import SHAPExplainableEngine
    from src.rag_engine import MedicalRAGEngine
    
    shap_eng = SHAPExplainableEngine()
    rag_eng = MedicalRAGEngine()
    cds_eng = ClinicalDecisionSupportEngine(provider="mock")
    
    test_patient = {"Gender": 0, "Hemoglobin": 9.0, "MCH": 21.5, "MCHC": 29.6, "MCV": 71.2}
    shap_res = shap_eng.predict_and_explain(test_patient)
    docs = rag_eng.retrieve_for_patient(shap_res)
    
    report = cds_eng.generate_report(test_patient, shap_res, docs)
    print("--- End-to-End CDS Report ---")
    print(report.encode('ascii', errors='ignore').decode('ascii'))
