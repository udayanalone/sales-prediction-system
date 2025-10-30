import pickle
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Print model information
print("Model type:", type(model).__name__)

# If it's an XGBoost model
if hasattr(model, 'get_booster'):
    print("Feature names:", model.get_booster().feature_names)
    print("Number of features:", len(model.get_booster().feature_names))
    
# If it's a scikit-learn pipeline or similar
if hasattr(model, 'feature_names_in_'):
    print("Feature names:", model.feature_names_in_)
    print("Number of features:", len(model.feature_names_in_))
