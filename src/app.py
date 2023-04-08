import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import numpy as np


@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load('facebookresearch/detectron2', 'faster_rcnn', pretrained=True,
                           model_name='COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
                           )
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def detect_nsfw(model, image):
    scores = model(image)[0]['scores'].detacg().cpu().numpy()
    nsfw_score = np.max(scores)
    return nsfw_score


def main():
    st.set_page_config(page_title='InsPyredGallery', page_icon='🔞', layout='wide')
    st.title('InsPyredGallery: An NSFW Image Scanner')

    model = load_model()

    image_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if image_file is not None:
        image = preprocess_image(image_file)
        nsfw_score = detect_nsfw(model, image)

        st.subheader('Original Image')
        st.image(image_file, use_column_width=True)

        if nsfw_score >= 0.5:
            st.warning('⚠️ This image contains NSFW content with a probability of {:.2f}%.'.format(nsfw_score * 100))
        else:
            st.success('✅ This image does not contain NSFW content with a probability of {:.2f}%.'.format(
                (1 - nsfw_score) * 100))

