from tensorflow.keras.models import load_model

model = load_model('model.h5')


def predict(image):
    """
    Predict the class of the given image using the pre-trained model.

    Args:
        image: The input image to be classified.

    Returns:
        The predicted class of the image.
    """
    # Preprocess the image as required by the model
    # For example, resizing, normalization, etc.

    # Make prediction
    prediction = model.predict(image)

    # Post-process the prediction if necessary
    return prediction


def load_model_from_file(file_path):
    """
    Load a pre-trained model from a file.

    Args:
        file_path: The path to the model file.

    Returns:
        The loaded model.
    """
    return load_model(file_path)

