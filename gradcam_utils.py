import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# те же нормализации, что при обучении
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


class GradCAM:
    def __init__(self, model, target_layer_name="features.denseblock4"):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name

        self.gradients = None
        self.activations = None

        # привязываем hook к нужному слою
        target_layer = self.get_target_layer()
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def get_target_layer(self):
        # для DenseNet121 последний блок features.denseblock4
        layer = self.model.features.denseblock4
        return layer

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(DEVICE)

        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # обнуляем градиенты и считаем их
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradients  # [N, C, H, W]
        activations = self.activations  # [N, C, H, W]

        # усреднение градиентов по пространственным осям
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # [N, C, 1, 1]

        # линейная комбинация активаций
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [N, 1, H, W]
        cam = cam.relu()

        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)  # нормализация [0,1]

        return cam


def load_image_as_tensor(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = preprocess(img).unsqueeze(0)  # [1, 3, 224, 224]
    return img, tensor


def overlay_cam_on_image(pil_img, cam, alpha=0.4):
    # PIL -> OpenCV (BGR)
    img = np.array(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    h, w, _ = img.shape
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

    # назад в RGB для matplotlib
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return overlay


def show_gradcam(pil_img, cam):
    overlay = overlay_cam_on_image(pil_img, cam)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(pil_img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
