
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO

# CONFIG

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224
STAGE1_THRESHOLD = 0.5

IDX_TO_CLASS = {
    0: 'Brain_Trans_cerebellum',
    1: 'Brain_Trans_thalamic',
    2: 'Brain_Trans_ventricular',
    3: 'Fetal_abdomen',
    4: 'Fetal_femur',
    5: 'Fetal_thorax',
    6: 'Maternal_cervix'
}

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# MODEL ARCHITECTURES

def build_stage1_model():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    return model

def build_stage2_model(num_classes=7):
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

# GRAD-CAM

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None, is_binary=False):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if is_binary:
            score = output[:, 0]
        else:
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            score = output[:, class_idx]

        score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


def overlay_heatmap(image_tensor, cam, alpha=0.45):
    mean = np.array(IMAGENET_MEAN).reshape(3, 1, 1)
    std = np.array(IMAGENET_STD).reshape(3, 1, 1)
    img = image_tensor.cpu().numpy()
    img = img * std + mean
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlayed = (1 - alpha) * img + alpha * heatmap
    overlayed = np.clip(overlayed, 0, 1)

    return img, heatmap, overlayed


def image_array_to_base64(img_array):
    img_uint8 = (img_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# MODEL LOADING 

def load_models(stage1_weights_path, stage2_weights_path):
    stage1_model = build_stage1_model().to(DEVICE)
    stage1_model.load_state_dict(torch.load(stage1_weights_path, map_location=DEVICE))
    stage1_model.eval()

    stage2_model = build_stage2_model(num_classes=7).to(DEVICE)
    stage2_model.load_state_dict(torch.load(stage2_weights_path, map_location=DEVICE))
    stage2_model.eval()

    gradcam_s1 = GradCAM(stage1_model, stage1_model.layer4[-1])
    gradcam_s2 = GradCAM(stage2_model, stage2_model.layer4[-1])

    return stage1_model, stage2_model, gradcam_s1, gradcam_s2

# MAIN PREDICTION FUNCTION

def predict_with_gradcam(image_path, stage1_model, stage2_model, gradcam_s1, gradcam_s2,
                          threshold=STAGE1_THRESHOLD):
    image = Image.open(image_path).convert('RGB')
    input_tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    result = {
        'final_label': None,
        'stage1_confidence': None,
        'stage2_confidences': None,
        'gradcam_overlay': None,
        'original_image': None
    }

    stage1_logit = stage1_model(input_tensor)
    stage1_prob = torch.sigmoid(stage1_logit).item()
    result['stage1_confidence'] = round(stage1_prob, 4)

    if stage1_prob < threshold:
        result['final_label'] = 'Other'
        cam = gradcam_s1.generate(input_tensor, is_binary=True)
        orig_img, heatmap, overlay = overlay_heatmap(input_tensor.squeeze(0), cam)
        result['gradcam_overlay'] = image_array_to_base64(overlay)
        result['original_image'] = image_array_to_base64(orig_img)
        return result

    stage2_logits = stage2_model(input_tensor)
    stage2_probs = torch.softmax(stage2_logits, dim=1).squeeze().detach().cpu().numpy()

    stage2_confidences = {IDX_TO_CLASS[i]: round(float(stage2_probs[i]), 4) for i in range(len(stage2_probs))}
    predicted_class_idx = int(stage2_probs.argmax())

    result['final_label'] = IDX_TO_CLASS[predicted_class_idx]
    result['stage2_confidences'] = stage2_confidences

    cam = gradcam_s2.generate(input_tensor, class_idx=predicted_class_idx, is_binary=False)
    orig_img, heatmap, overlay = overlay_heatmap(input_tensor.squeeze(0), cam)
    result['gradcam_overlay'] = image_array_to_base64(overlay)
    result['original_image'] = image_array_to_base64(orig_img)

    return result
