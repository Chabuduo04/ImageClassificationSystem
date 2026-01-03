import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models import SimpleCNN, get_resnet18, get_mobilenet_v2

# ---------- 配置 ----------
st.set_page_config(page_title="Image Classification System")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MODEL_NAME = "cnn"   # cnn | resnet | mobilenet
MODEL_PATH = f"checkpoints/{MODEL_NAME}.pth"

# ---------- 模型加载 ----------
@st.cache_resource
def load_model():
    if MODEL_NAME == "cnn":
        model = SimpleCNN(num_classes=10)
    elif MODEL_NAME == "resnet":
        model = get_resnet18(num_classes=10)
    elif MODEL_NAME == "mobilenet":
        model = get_mobilenet_v2(num_classes=10)
    else:
        raise ValueError("Unknown model")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ---------- 预处理 ----------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# ---------- 推理函数 ----------
def predict(image: Image.Image):
    img = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(img)
        prob = F.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)
    return CLASS_NAMES[pred.item()], conf.item()

# ---------- UI ----------
st.title("🧠 基于深度学习的图像分类系统")
st.write("支持单张图片与多张图片批量预测")

uploaded_files = st.file_uploader(
    "上传图片（支持多张）",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    st.subheader("预测结果")
    cols = st.columns(3)

    for idx, file in enumerate(uploaded_files):
        image = Image.open(file).convert("RGB")
        label, confidence = predict(image)

        with cols[idx % 3]:
            st.image(image, use_column_width=True)
            st.markdown(f"**预测类别：** {label}")
            st.markdown(f"**置信度：** {confidence:.2f}")
