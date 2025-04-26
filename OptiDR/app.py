import cv2
import numpy as np
import torch
from torchvision import models, transforms
import matplotlib.pyplot as plt
import streamlit as st
import time
from streamlit.components.v1 import html

# Load pre-trained ResNeXt model
@st.cache_resource
def load_model():
    model = models.resnext101_32x8d(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 5)
    model.load_state_dict(torch.load('OptiDR/model.pt', map_location=torch.device('cpu')))
    return model.eval()

model = load_model()

# Preprocessing
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grad-CAM for PyTorch
def grad_cam(image, model, target_layer='layer4'):
    model.eval()
    img_tensor = preprocess(image).unsqueeze(0)
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_module = dict(model.named_modules())[target_layer]
    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    class_idx = output.argmax(dim=1).item()
    model.zero_grad()
    output[0, class_idx].backward()

    activation = activations[0].detach()
    gradient = gradients[0].detach()
    pooled_grad = gradient.mean(dim=(2, 3), keepdim=True)
    weighted = activation * pooled_grad
    heatmap = weighted.sum(dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap = heatmap.cpu().numpy()
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    forward_handle.remove()
    backward_handle.remove()

    return heatmap

# Custom CSS for light medical retina theme

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;600&display=swap');
    * {
        font-family: 'Rubik', sans-serif;
    }
    .stApp {
        background-image: url('https://upload.wikimedia.org/wikipedia/commons/4/41/Retinal_scan_example.jpg');
        background-size: cover;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
        backdrop-filter: blur(4px);
    }
    .header, .feature-card, .accuracy-box {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        color: #0a3d62;
    }
    .highlight {
        background-color: #dff9fb;
        padding: 12px;
        border-left: 5px solid #0984e3;
        border-radius: 8px;
        font-weight: 600;
        color: #2d3436;
    }
    .accuracy-box {
        border-left: 6px solid #00b894;
    }
    h1, h2, h3, h4 {
        color: #0984e3;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Diagnostic", "ℹ️ About Retinopathy", "📘 DR Classification Info", "📂 Dataset & Model Info"])
    inject_custom_css()
    inject_custom_css()
    with tab1:
        st.title("OptiDR")
        st.markdown("""
        <div class='header'>
            <h1>Your Digital Eye Wellness Companion</h1>
            <p class='highlight'>Upload a retinal scan and let our AI assist with Diabetic Retinopathy detection using state-of-the-art visualization and classification.</p>
        </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a retinal scan", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            img_tensor = preprocess(img).unsqueeze(0)
            pred = model(img_tensor)
            pred_class = torch.argmax(pred).item()

        heatmap = grad_cam(img, model, target_layer='layer4')
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Retinal Scan", use_container_width=True)
        with col2:
            st.image(superimposed_img, caption="AI Heatmap Visualization", use_container_width=True)

        if pred_class >= 2:
            class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
            st.error(f"Prediction: {class_labels[pred_class]} (Class {pred_class}) | Referral Urgency: High")
            st.markdown("""
            <div class='highlight'>
            ⚠️ **Important:** Signs of moderate to severe diabetic retinopathy detected.
            
            **Next Steps:**
            - Consult a licensed ophthalmologist promptly.
            - Avoid delay in scheduling an in-person retinal evaluation.
            - Share these scan results with your doctor.
            
            Early action can preserve vision and prevent complications.
            </div>
            """, unsafe_allow_html=True)
        else:
            class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
            st.success(f"Prediction: {class_labels[pred_class]} (Class {pred_class}) | Referral Urgency: Low")
            st.markdown("""
            <div class='highlight'>
            ✅ **Good news!** No immediate signs of concerning diabetic retinopathy detected.
            
            **Recommended Actions:**
            - Maintain regular eye checkups (at least once a year).
            - Continue healthy lifestyle habits for diabetes management.
            - Keep monitoring for any changes in vision.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='accuracy-box'>
            <h4>Model Performance Snapshot 📊</h4>
            <p>• Accuracy: 89.4%</p>
            <p>• Precision: 88.1%</p>
            <p>• Recall: 87.3%</p>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.header("What is Diabetic Retinopathy?")
        st.markdown("""
        Diabetic retinopathy is a complication of diabetes that affects the eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina).
        
        Early detection through retinal imaging and AI-based diagnostics like RetinaGuard AI can help prevent vision loss.
        
        **Key symptoms:** blurry vision, floaters, dark areas of vision, and vision loss.
        """)
        st.image("https://neoretina.com/blog/wp-content/uploads/2018/12/diabetic-rethonopaty.jpg", caption="Structure of the Retina Diagram", use_container_width=True)

    with tab3:
        st.header("📘 Diabetic Retinopathy (DR) Classification Stages")
        st.markdown("""
        | Class | Name               | Description |
        |-------|--------------------|-------------|
        | 0     | **No DR**           | No signs of diabetic retinopathy. Retinal blood vessels are healthy. |
        | 1     | **Mild**            | Microaneurysms may be present. Small areas of balloon-like swelling in the retina's blood vessels. |
        | 2     | **Moderate**        | Blood vessels that nourish the retina are blocked. More noticeable retinal damage. |
        | 3     | **Severe**          | Many more blood vessels are blocked, depriving the retina of its blood supply. |
        | 4     | **Proliferative DR**| Advanced stage with new abnormal blood vessels forming, which can leak and cause vision loss. |
        """)

    with tab4:
        st.header("📂 Dataset, Model, and Medical Insight")
        st.markdown("""
        ### 🧾 APTOS Blindness Detection Dataset
        The dataset used is the **APTOS 2019 Blindness Detection** dataset provided by Kaggle.
        It contains high-resolution retinal fundus images labeled across 5 stages of diabetic retinopathy:

        - 0: No DR
        - 1: Mild
        - 2: Moderate
        - 3: Severe
        - 4: Proliferative DR

        📌 [View the dataset on Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)

        ---
        ### 🧠 Transfer Learning Model: ResNeXt-101 32x8d
        We used **ResNeXt-101 32x8d**, a powerful deep convolutional neural network that improves on ResNet by grouping convolutions.

        - Pretrained on ImageNet
        - Fine-tuned for multi-class classification of DR levels
        - Integrated with Grad-CAM for model explainability

        📌 [ResNeXt Paper - Aggregated Residual Transformations](https://arxiv.org/abs/1611.05431)

        ---
        ### 🔬 Medical Indicators for Diabetic Retinopathy
        Detection is often based on clinical signs in fundus images:

        - **Microaneurysms** – early signs of DR, small red dots
        - **Hard Exudates** – yellowish-white lipid residues
        - **Cotton Wool Spots** – fluffy white patches, nerve fiber layer infarctions
        - **Hemorrhages** – dark red spots indicating bleeding
        - **Neovascularization** – hallmark of proliferative DR, new vessel formation

        📌 [Study: Detection of Diabetic Retinopathy using Fundus Images](https://www.ncbi.nlm.nih.gov/books/NBK560805/)
        """)
if __name__ == "__main__":
    main()
    st.markdown("<div style='text-align: center; padding: 1rem; font-size: 0.85rem; color: gray;'>© 2025 Team Nexus. All rights reserved.</div>", unsafe_allow_html=True)
    
