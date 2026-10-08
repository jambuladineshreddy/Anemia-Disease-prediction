import os
import sys
import joblib
import pandas as pd
import numpy as np
import shap

# Ensure project root is on sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODEL_PATH, FEATURE_NAMES, REFERENCE_RANGES


class SHAPExplainableEngine:
    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.feature_names = FEATURE_NAMES

    def predict_and_explain(self, patient_dict):
        """
        Takes patient lab metrics dictionary:
        e.g., {'Gender': 0, 'Hemoglobin': 9.0, 'MCH': 21.5, 'MCHC': 29.6, 'MCV': 71.2}
        Returns prediction, probability, local SHAP attributions, morphological subtype,
        risk tier, and borderline gray zone status.
        """
        input_df = pd.DataFrame([patient_dict])[self.feature_names]
        
        # 1. Machine Learning Prediction
        pred_class = int(self.model.predict(input_df)[0])
        pred_proba = float(self.model.predict_proba(input_df)[0][1])  # Probability of Anemia
        
        # 2. Local SHAP Values
        shap_values = self.explainer(input_df)
        
        # Binary classification SHAP values (Class 1 - Anemia)
        if len(shap_values.shape) == 3:
            sample_shap = shap_values.values[0, :, 1]
            base_value = float(shap_values.base_values[0, 1])
        elif len(shap_values.shape) == 2:
            sample_shap = shap_values.values[0]
            base_value = float(shap_values.base_values[0])
        else:
            sample_raw = self.explainer.shap_values(input_df)
            if isinstance(sample_raw, list):
                sample_shap = sample_raw[1][0]
                base_value = float(self.explainer.expected_value[1])
            else:
                sample_shap = sample_raw[0]
                base_value = float(self.explainer.expected_value)

        # 3. Format Feature Attributions
        feature_attributions = []
        for feat, val, s_val in zip(self.feature_names, input_df.iloc[0], sample_shap):
            impact = "Increases Anemia Risk (+)" if s_val > 0 else "Protective / Decreases Risk (-)"
            feature_attributions.append({
                "feature": feat,
                "value": float(val),
                "shap_value": float(s_val),
                "impact": impact,
                "abs_shap": abs(float(s_val))
            })
            
        # Sort by magnitude of impact
        feature_attributions.sort(key=lambda x: x["abs_shap"], reverse=True)

        # 4. Morphological Categorization based on MCV & Hemoglobin
        gender_str = "Male" if patient_dict.get("Gender", 1) == 1 else "Female"
        mcv_val = float(patient_dict.get("MCV", 85.0))
        hb_val = float(patient_dict.get("Hemoglobin", 13.0))
        
        if mcv_val < 80.0:
            morphology = "Microcytic Hypochromic"
        elif mcv_val > 100.0:
            morphology = "Macrocytic"
        else:
            morphology = "Normocytic Normochromic"

        # 5. Clinical Risk Tier & Borderline Gray Zone Detection
        if pred_proba < 0.25:
            risk_tier = "Low Anemia Probability"
            risk_severity = "low"
        elif pred_proba < 0.65:
            risk_tier = "Borderline / Moderate Risk (Physiological Gray Zone)"
            risk_severity = "moderate"
        elif pred_proba < 0.88:
            risk_tier = "High Anemia Risk"
            risk_severity = "high"
        else:
            risk_tier = "Severe Anemia Profile"
            risk_severity = "severe"

        # Gray Zone Check (Females 11.5 - 12.5, Males 12.5 - 13.5)
        gray_zone = False
        if (gender_str == "Female" and 11.5 <= hb_val <= 12.5) or (gender_str == "Male" and 12.5 <= hb_val <= 13.5):
            gray_zone = True

        # Check reference anomalies
        anomalies = []
        hb_cutoff = REFERENCE_RANGES["Hemoglobin"]["cutoff_anemia"][gender_str]
        if hb_val < hb_cutoff:
            anomalies.append(f"Low Hemoglobin ({hb_val} g/dL < cutoff {hb_cutoff} g/dL)")
        if mcv_val < 80.0:
            anomalies.append(f"Low MCV ({mcv_val} fL < 80.0 fL - Microcytosis)")
        elif mcv_val > 100.0:
            anomalies.append(f"High MCV ({mcv_val} fL > 100.0 fL - Macrocytosis)")
            
        mch_val = float(patient_dict.get("MCH", 28.0))
        if mch_val < 27.0:
            anomalies.append(f"Low MCH ({mch_val} pg < 27.0 pg - Hypochromasia)")
            
        mchc_val = float(patient_dict.get("MCHC", 33.0))
        if mchc_val < 32.0:
            anomalies.append(f"Low MCHC ({mchc_val} g/dL < 32.0 g/dL)")

        return {
            "prediction": "Anemic" if pred_class == 1 else "Non-Anemic",
            "prediction_code": pred_class,
            "probability": round(pred_proba, 4),
            "probability_percent": f"{round(pred_proba * 100, 1)}%",
            "base_value": round(base_value, 4),
            "morphology": morphology,
            "gender": gender_str,
            "risk_tier": risk_tier,
            "risk_severity": risk_severity,
            "is_gray_zone": gray_zone,
            "anomalies": anomalies,
            "feature_attributions": feature_attributions,
            "raw_input": patient_dict
        }


if __name__ == "__main__":
    engine = SHAPExplainableEngine()
    test_patient = {"Gender": 0, "Hemoglobin": 9.0, "MCH": 21.5, "MCHC": 29.6, "MCV": 71.2}
    res = engine.predict_and_explain(test_patient)
    print("--- SHAP Engine Output ---")
    print(f"Prediction: {res['prediction']} ({res['probability_percent']})")
    print(f"Risk Tier: {res['risk_tier']}")
    print(f"Morphology: {res['morphology']}")
    print(f"Top Risk Drivers: {res['feature_attributions'][:2]}")
