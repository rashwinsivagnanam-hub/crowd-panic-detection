import torch
import cv2
from torchvision import transforms, models
from torch import nn

# ----------------------------
# CLASS NAMES
# ----------------------------
classes = ['normal', 'panic_escape', 'violence']

# ----------------------------
# LOAD MODEL
# ----------------------------
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)

model.load_state_dict(torch.load("crowd_behavior_model.pth"))
model.eval()

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ----------------------------
# VIDEO PATH
# ----------------------------
video_path = "test_video1.mp4"

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # preprocess frame
    img = transform(frame)
    img = img.unsqueeze(0)

    # prediction
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs,1)

    label = classes[predicted.item()]

    # show label on video
    cv2.putText(frame, label,
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Crowd Behavior Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()