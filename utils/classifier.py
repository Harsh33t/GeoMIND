import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def coordinate_to_pixel(lat, lon, transform, width, height):
    """
    Translates WGS84 geographic coordinates (lat, lon) to pixel coordinates (row, col)
    using the affine transform.
    """
    # Invert transform: pixel x, y = ~transform * (lon, lat)
    col, row = ~transform * (lon, lat)
    col, row = int(round(col)), int(round(row))
    
    # Ensure they are within image bounds
    if 0 <= col < width and 0 <= row < height:
        return row, col
    return None

def extract_features_at_points(bands, indices, training_points, transform, width, height):
    """
    Extracts 9 features (6 bands + 3 indices) for each training point coordinate.
    """
    features_list = []
    labels = []
    classes = []
    
    # Feature keys in order
    feature_names = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'iron_oxide', 'clay_minerals', 'ferrous_iron']
    
    for lat, lon, label, class_name in training_points:
        pixel_coords = coordinate_to_pixel(lat, lon, transform, width, height)
        if pixel_coords is None:
            continue
        
        row, col = pixel_coords
        
        # Gather pixel feature vector
        vector = []
        for b in ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']:
            vector.append(bands[b][row, col])
        for idx in ['iron_oxide', 'clay_minerals', 'ferrous_iron']:
            vector.append(indices[idx][row, col])
            
        features_list.append(vector)
        labels.append(label)
        classes.append(class_name)
        
    X = np.array(features_list)
    y = np.array(labels)
    
    return X, y, classes, feature_names

def train_mineral_classifier(X, y, feature_names, class_names):
    """
    Trains a Random Forest Classifier to identify mineral signatures.
    Returns the trained classifier, out-of-bag accuracy score, and feature importances.
    """
    if len(X) < 5:
        raise ValueError("Insufficient training points. Please add at least 5 points to train the Random Forest.")
        
    # Standardize classes present
    unique_y = np.unique(y)
    
    # Train Random Forest Classifier with Out-Of-Bag evaluation enabled
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        oob_score=True,
        class_weight='balanced'
    )
    clf.fit(X, y)
    
    oob_score = clf.oob_score_
    
    # Calculate feature importances
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Generate classification report on training set as self-consistency check
    y_pred = clf.predict(X)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    
    return clf, oob_score, importance_df, report

def predict_mineral_probabilities(clf, bands, indices, width, height):
    """
    Predicts mineral class probabilities for every pixel in the scene.
    Returns:
      1. Multi-class prediction map [H, W] containing the class label with highest probability.
      2. Dictionary mapping class label to its continuous probability spatial map [H, W].
    """
    # 1. Flatten all 9 spatial feature layers to [N_pixels, 9] shape
    n_pixels = width * height
    X_predict = np.zeros((n_pixels, 9), dtype=np.float32)
    
    feature_layers = [
        bands['B02'], bands['B03'], bands['B04'], bands['B08'], bands['B11'], bands['B12'],
        indices['iron_oxide'], indices['clay_minerals'], indices['ferrous_iron']
    ]
    
    for i, layer in enumerate(feature_layers):
        X_predict[:, i] = layer.ravel()
        
    # 2. Perform batched prediction
    probabilities = clf.predict_proba(X_predict) # Shape: [N_pixels, N_classes]
    predictions = clf.predict(X_predict)         # Shape: [N_pixels]
    
    # 3. Reshape multi-class classification map to original [H, W]
    classification_map = predictions.reshape((height, width))
    
    # 4. Map class labels to continuous probability rasters
    probability_maps = {}
    for i, class_label in enumerate(clf.classes_):
        class_prob = probabilities[:, i].reshape((height, width))
        probability_maps[int(class_label)] = class_prob
        
    return classification_map, probability_maps
