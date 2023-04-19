import numpy as np
import typing 
import numba
import numpy.typing as npt
from skimage.transform import resize

@numba.jit
def preprocess_img(img : npt.NDArray[typing.Any]) -> npt.NDArray[typing.Any]:

    '''
    Image to numpy, Resize
    
    Args:
        img: np.ndarray

    Returns:
        img: np.ndarray
    '''

    img = np.array(img).astype(np.float32)

    IMG_HEIGHT = 96
    IMG_WIDTH = 96
    
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode="constant", preserve_range=True) 

    return img
    

@numba.jit
def array_modelinput(img: npt.NDArray[typing.Any]) -> npt.NDArray[typing.Any]:
    '''
    Image to numpy, Resize
    
    Args:
        img: np.ndarray

    Returns:
        img: np.ndarray
    '''
        
    img = np.array(img).astype(np.float32)
    
    img_arr = np.expand_dims(img, axis=0)
    return img_arr