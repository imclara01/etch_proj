import shap
import numpy as np
import pandas as pd

class SHAPExplainer:
    def __init__(self, lgb_model, features):
        self.explainer = shap.TreeExplainer(lgb_model)
        self.features = features
        
    def explain(self, scaled_features, pred_idx):
        """
        Explains the prediction for a specific class index.
        Returns top 3 contributing sensors.
        """
        shap_vals = self.explainer.shap_values(scaled_features)
        
        # Handle different SHAP output formats (list for multiclass or 3D array)
        if isinstance(shap_vals, list):
            # shap_vals[class_index][sample_index, feature_index]
            class_shap = shap_vals[pred_idx][0]
        else:
            # Some versions return (samples, features, classes)
            if len(shap_vals.shape) == 3:
                class_shap = shap_vals[0, :, pred_idx]
            else:
                class_shap = shap_vals[0]
                
        # Get top 3 sensors by absolute SHAP value
        top_indices = np.argsort(np.abs(class_shap))[-3:][::-1]
        top_sensors = [self.features[i] for i in top_indices]
        
        # Create a dictionary of sensor names and their contributions
        contributions = {self.features[i]: float(class_shap[i]) for i in top_indices}
        
        return top_sensors, contributions

if __name__ == "__main__":
    from inference import InferenceEngine
    engine = InferenceEngine()
    explainer = SHAPExplainer(engine.lgb_model, engine.features)
    
    test_df = pd.read_csv('data/test_split.csv')
    sample = test_df.iloc[0].to_dict()
    result = engine.predict(sample)
    
    if result['status'] != 'Normal':
        # Need to find the index of the predicted label
        pred_idx = list(engine.le.classes_).index(result['predicted_label'])
        sensors, contribs = explainer.explain(result['scaled_features'], pred_idx)
        print(f"Top Sensors for {result['predicted_label']}: {sensors}")
        print(f"Contributions: {contribs}")
