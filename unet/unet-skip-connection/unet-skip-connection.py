import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    print(decoder_features.shape)
    h = (encoder_features.shape[1]-decoder_features.shape[1])//2
    w = (encoder_features.shape[2]-decoder_features.shape[2])//2
    cropped = encoder_features[:, h:h+decoder_features.shape[1], w:w+decoder_features.shape[2],:]
    concat = np.concatenate((cropped, decoder_features), axis=3)
    return concat
