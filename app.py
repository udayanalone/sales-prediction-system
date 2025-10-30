import os
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Get the directory where this script is located
basedir = os.path.abspath(os.path.dirname(__file__))

# Create the Flask app
app = Flask(__name__)

# Define data for dropdowns
item_types = [
    'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
    'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Canned',
    'Deli', 'Breads', 'Hard Drinks', 'Starchy Foods', 'Seafood', 'Others'
]

outlet_types = [
    'Grocery Store',
    'Supermarket Type1',
    'Supermarket Type2',
    'Supermarket Type3'
]

outlet_sizes = ['Small', 'Medium', 'High']
location_types = ['Tier 1', 'Tier 2', 'Tier 3']

# Initialize label encoders
label_encoders = {}

def encode_value(feature, value):
    """Encode a single categorical value using the appropriate label encoder."""
    if feature not in label_encoders:
        label_encoders[feature] = LabelEncoder()
        if feature == 'Item_Type':
            label_encoders[feature].fit(item_types)
        elif feature == 'Outlet_Type':
            label_encoders[feature].fit(outlet_types)
        elif feature == 'Outlet_Size':
            label_encoders[feature].fit(outlet_sizes)
        elif feature == 'Outlet_Location_Type':
            label_encoders[feature].fit(location_types)
        elif feature == 'Item_Fat_Content':
            label_encoders[feature].fit(['Low Fat', 'Regular'])
    
    # Transform the value
    return label_encoders[feature].transform([value])[0]

# Load or train model
def load_model():
    model_path = os.path.join(basedir, 'model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# Verify model is loaded and has the expected type
if model is not None:
    print(f"Model loaded successfully. Model type: {type(model).__name__}")
    if hasattr(model, 'feature_importances_'):
        print(f"Model has {len(model.feature_importances_)} features")
else:
    print("Warning: Could not load the model. Using dummy predictions.")

# Routes
@app.route('/')
def home():
    # Debug: Print the template directory
    template_path = os.path.join(app.root_path, 'templates')
    print(f"Template directory: {template_path}")
    print(f"Template exists: {os.path.exists(os.path.join(template_path, 'index.html'))}")
    
    try:
        print("Files in template directory:")
        for f in os.listdir(template_path):
            print(f"- {f}")
    except Exception as e:
        print(f"Error listing template directory: {e}")
    
    return render_template('index.html',
                         item_types=item_types,
                         outlet_types=outlet_types,
                         outlet_sizes=outlet_sizes,
                         location_types=location_types)

def validate_input_data(data):
    """Validate input data and return error message if validation fails."""
    required_fields = {
        'item_identifier': str,
        'item_weight': (float, "must be a number"),
        'item_fat_content': str,
        'item_visibility': (float, "must be a number between 0 and 1"),
        'item_type': str,
        'item_mrp': (float, "must be a positive number"),
        'outlet_identifier': (int, "must be an integer"),
        'outlet_establishment_year': (int, "must be a year between 1985 and 2025"),
        'outlet_size': str,
        'outlet_location_type': str,
        'outlet_type': str
    }
    
    errors = []
    validated_data = {}
    
    for field, field_type in required_fields.items():
        value = data.get(field)
        
        # Check if field is missing
        if value is None or value == '':
            errors.append(f"{field.replace('_', ' ').title()} is required")
            continue
            
        # Check field type and constraints
        if isinstance(field_type, tuple):
            type_func, error_msg = field_type
            try:
                value = type_func(value)
                # Additional validation for specific fields
                if field == 'item_visibility' and not (0 <= value <= 1):
                    errors.append(f"Item visibility {error_msg}")
                elif field == 'item_mrp' and value <= 0:
                    errors.append(f"Item MRP {error_msg}")
                elif field == 'outlet_establishment_year' and not (1985 <= value <= 2025):
                    errors.append(f"Outlet establishment year {error_msg}")
                validated_data[field] = value
            except (ValueError, TypeError):
                errors.append(f"{field.replace('_', ' ').title()} {error_msg}")
        else:
            validated_data[field] = field_type(value)
    
    return validated_data, errors

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get and validate form data
        data = request.form
        validated_data, errors = validate_input_data(data)
        
        if errors:
            return jsonify({
                'success': False,
                'error': 'Validation Error',
                'details': errors
            }), 400
            
        # Prepare input data for prediction with proper types
        input_data = {
            'Item_Identifier': validated_data['item_identifier'],
            'Item_Weight': validated_data['item_weight'],
            'Item_Fat_Content': validated_data['item_fat_content'],
            'Item_Visibility': validated_data['item_visibility'],
            'Item_Type': validated_data['item_type'],
            'Item_MRP': validated_data['item_mrp'],
            'Outlet_Identifier': validated_data['outlet_identifier'],
            'Outlet_Establishment_Year': validated_data['outlet_establishment_year'],
            'Outlet_Size': validated_data['outlet_size'],
            'Outlet_Location_Type': validated_data['outlet_location_type'],
            'Outlet_Type': validated_data['outlet_type']
        }
        
        # Encode categorical features
        input_data['Item_Fat_Content'] = encode_value('Item_Fat_Content', input_data['Item_Fat_Content'])
        input_data['Item_Type'] = encode_value('Item_Type', input_data['Item_Type'])
        input_data['Outlet_Size'] = encode_value('Outlet_Size', input_data['Outlet_Size'])
        input_data['Outlet_Location_Type'] = encode_value('Outlet_Location_Type', input_data['Outlet_Location_Type'])
        input_data['Outlet_Type'] = encode_value('Outlet_Type', input_data['Outlet_Type'])
        
        # Create feature array in the correct order expected by the model
        features = [
            input_data['Item_Identifier'],
            input_data['Item_Weight'],
            input_data['Item_Fat_Content'],
            input_data['Item_Visibility'],
            input_data['Item_Type'],
            input_data['Item_MRP'],
            input_data['Outlet_Identifier'],
            input_data['Outlet_Establishment_Year'],
            input_data['Outlet_Size'],
            input_data['Outlet_Location_Type'],
            input_data['Outlet_Type']
        ]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Ensure prediction is a positive number
        prediction = max(0, float(prediction))
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'formatted_prediction': f'${prediction:,.2f}'
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid input data',
            'details': str(e)
        }), 400
    except KeyError as e:
        return jsonify({
            'success': False,
            'error': 'Missing required field',
            'field': str(e).strip("'")
        }), 400
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during prediction: {str(e)}\n{error_details}")
        return jsonify({
            'success': False,
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
