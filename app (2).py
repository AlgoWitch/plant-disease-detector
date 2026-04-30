import gradio as gr
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import json

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

num_classes = len(class_names)
device = torch.device("cpu")
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load('plant_disease_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image):
    if image is None:
        return {}
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top5_probs, top5_idxs = torch.topk(probs, 5)
    return {class_names[i]: float(top5_probs[j])
            for j, i in enumerate(top5_idxs)}

css = """
* { box-sizing: border-box; }
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
    padding: 40px 24px !important;
    background-color: #f5f7f2 !important;
}
.gradio-container, .gradio-container * {
    color-scheme: light !important;
}
.dark .gradio-container {
    background-color: #f5f7f2 !important;
}
#header { margin-bottom: 28px; }
#header h1 { font-size: 24px; font-weight: 500; color: #2d3a2e !important; margin-bottom: 6px; }
#header p { font-size: 14px; color: #4a5e4b !important; line-height: 1.6; }
#crops-label {
    font-size: 11px;
    color: #4a5e4b !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 20px 0 10px;
}
#crops-list { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 28px; }
.crop-tag {
    font-size: 12px;
    color: #2d4a2e !important;
    background: #e8f0e8 !important;
    border: 0.5px solid #c8ddc9 !important;
    border-radius: 20px;
    padding: 3px 12px;
}
.block, .wrap, [data-testid="image"], .label-wrap {
    background-color: #ffffff !important;
    border-color: #c8ddc9 !important;
    color: #2d3a2e !important;
}
span[data-testid="block-label"], .block-label, label {
    color: #4a5e4b !important;
    background: transparent !important;
}
button.primary, button[variant="primary"] {
    background-color: #d4e8d4 !important;
    color: #2d4a2e !important;
    border: 1px solid #b8d4b8 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
}
button.primary:hover {
    background-color: #c2dbc2 !important;
}
button.secondary, button[variant="secondary"] {
    background-color: #ffffff !important;
    color: #4a5e4b !important;
    border: 1px solid #c8ddc9 !important;
    border-radius: 10px !important;
    font-size: 14px !important;
}
button.secondary:hover {
    background-color: #f0f5f0 !important;
}
#stats-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 20px;
}
.stat-box {
    background: #ffffff !important;
    border: 1px solid #c8ddc9;
    border-radius: 12px;
    padding: 16px 20px;
}
.stat-num {
    font-size: 22px;
    font-weight: 500;
    color: #2d3a2e !important;
}
.stat-lbl {
    font-size: 12px;
    color: #4a5e4b !important;
    margin-top: 2px;
}
footer { display: none !important; }
.label-wrap span,
[data-testid="label-output"] span,
.output-class,
.confidence-set span,
span.text-token {
    color: #2d3a2e !important;
}
.confidence-set .label {
    color: #2d3a2e !important;
}
.confidence-set .confidence-number {
    color: #4a5e4b !important;
}
.output-class {
    color: #2d3a2e !important;
    font-size: 16px !important;
    font-weight: 500 !important;
}
"""

with gr.Blocks(css=css, title="Plant Disease Detector") as demo:
    gr.HTML("""
    <div id='header'>
        <h1>Plant Disease Detector</h1>
        <p>Upload a leaf photo to identify diseases across 14 crop types.<br>
        Trained on 54,000+ images from the PlantVillage dataset.</p>
    </div>
    <div id='crops-label'>Supported crops</div>
    <div id='crops-list'>
        <span class='crop-tag'>Apple</span>
        <span class='crop-tag'>Blueberry</span>
        <span class='crop-tag'>Cherry</span>
        <span class='crop-tag'>Corn</span>
        <span class='crop-tag'>Grape</span>
        <span class='crop-tag'>Orange</span>
        <span class='crop-tag'>Peach</span>
        <span class='crop-tag'>Bell Pepper</span>
        <span class='crop-tag'>Potato</span>
        <span class='crop-tag'>Raspberry</span>
        <span class='crop-tag'>Soybean</span>
        <span class='crop-tag'>Squash</span>
        <span class='crop-tag'>Strawberry</span>
        <span class='crop-tag'>Tomato</span>
    </div>
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Leaf photo", height=300)
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("Analyze leaf", variant="primary")
        with gr.Column():
            output = gr.Label(num_top_classes=5, label="Diagnosis")

    gr.HTML("""
    <div id='stats-row'>
        <div class='stat-box'>
            <div class='stat-num'>38</div>
            <div class='stat-lbl'>Disease classes</div>
        </div>
        <div class='stat-box'>
            <div class='stat-num'>99%</div>
            <div class='stat-lbl'>Validation accuracy</div>
        </div>
        <div class='stat-box'>
            <div class='stat-num'>14</div>
            <div class='stat-lbl'>Crop types</div>
        </div>
    </div>
    """)

    submit_btn.click(fn=predict, inputs=image_input, outputs=output)
    clear_btn.click(fn=lambda: (None, {}), inputs=[], outputs=[image_input, output])

demo.launch()
