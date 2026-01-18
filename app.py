# ============================================================
# Diabetic Retinopathy Detection Streamlit Web App (FIXED)
# ============================================================

import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="DR Detection System",
    layout="wide",
    page_icon="🩺"
)

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from PIL import Image
import gdown
gdown.download( id= "1PZK8n3Dv4G-qWn0gDML5uujEfkMznAxi", output="model_Eyepaycs.keras", quiet=False)


# ============================================================
# Custom ViT Layers
# ============================================================
@tf.keras.utils.register_keras_serializable()
class PatchExtractor(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = tf.shape(patches)[-1]
        return tf.reshape(patches, [tf.shape(images)[0], -1, patch_dims])

    def get_config(self):
        return {"patch_size": self.patch_size}


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)

    def build(self, input_shape):
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.projection.units
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        return {"num_patches": self.num_patches}

# ============================================================
# Load Model
# ============================================================
@st.cache_resource
def load_dr_model():
    return load_model(
        "model_Eyepaycs.keras",
        custom_objects={
            "PatchExtractor": PatchExtractor,
            "PatchEncoder": PatchEncoder
        },
        compile=False
    )

model = load_dr_model()

# ============================================================
# Constants
# ============================================================
CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]

MEDICINE_RECOMMENDATION = {
    "No_DR": ["No medication required", "Routine eye check-up"],
    "Mild": ["Antioxidant supplements", "Vitamin B-complex"],
    "Moderate": ["Anti-VEGF injections", "Blood sugar control"],
    "Severe": ["Anti-VEGF therapy", "Steroid injections"],
    "Proliferative_DR": ["Laser photocoagulation", "Vitrectomy surgery"]
}

# ============================================================
# Preprocessing (RETURNS TENSOR)
# ============================================================
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return tf.convert_to_tensor(image, dtype=tf.float32)

# ============================================================
# ViT-Safe Saliency Map (FIXED)
# ============================================================
def generate_saliency_map(image_tensor):
    image_tensor = tf.cast(image_tensor, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        predictions = model(image_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    gradients = tape.gradient(loss, image_tensor)
    saliency = tf.reduce_mean(tf.abs(gradients), axis=-1)

    saliency = saliency[0].numpy()
    saliency = (saliency - saliency.min()) / (saliency.max() + 1e-8)
    return saliency

# ============================================================
# UI
# ============================================================
st.title("🩺 Diabetic Retinopathy Detection & Recommendation System")
st.markdown("Upload a **fundus image** to predict DR stage and view saliency map.")

uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = preprocess_image(image)

    preds = model.predict(img_tensor)
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))
    class_name = CLASS_NAMES[class_id]

    saliency = generate_saliency_map(img_tensor)
    saliency = cv2.resize(saliency, image.size)
    saliency = cv2.applyColorMap(np.uint8(255 * saliency), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(image), 0.6, saliency, 0.4, 0)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    with col2:
        st.image(
            overlay,
            caption=f"Predicted: {class_name} | Confidence: {confidence:.4f}",
            use_column_width=True
        )

    st.subheader("🧠 Prediction")
    st.write(f"**Class:** {class_name}")
    st.write(f"**Confidence:** {confidence:.4f}")

    st.subheader("💊 Recommended Treatment")
    for med in MEDICINE_RECOMMENDATION[class_name]:
        st.write(f"- {med}")

    st.warning("⚠️ Research & educational use only. Consult an ophthalmologist.")
