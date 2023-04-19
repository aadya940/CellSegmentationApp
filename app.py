import streamlit as st
from package.src.preprocessor import preprocess_img, array_modelinput
from package.src.model_loader import load_model
import cv2
import numpy as np

def ops_(img):
    img_ = preprocess_img(img)
    img_ = array_modelinput(img_)
    mod_ = load_model()
    return img_, mod_

def main():
    st.title('Image Segmentation with U-Net')
    st.write('Please upload the images below')
    uploaded_file = st.file_uploader("Upload Images of Cells", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        images = cv2.imdecode(file_bytes, 1)
    if st.button("Segment"):
        image, model = ops_(images)
        preds = model.predict(image)
        st.image(preds[0], "Mask of Cells Segmented")


if __name__ == "__main__":
    main()