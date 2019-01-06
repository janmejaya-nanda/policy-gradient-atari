import numpy as np
from PIL import Image

from constants import INPUT_SHAPE


def preprocess_observation(obs):
    # Convert to gray-scale and resize it
    image = Image.fromarray(obs, 'RGB').convert('L').resize((INPUT_SHAPE[0], INPUT_SHAPE[1]))
    # Convert image to array and return it
    return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1], image.size[0])


def get_next_state(current, obs):
    # Next state is composed by the last n-1 images of the previous state and new observation
    return np.append(current[1:], [obs], axis=0)
