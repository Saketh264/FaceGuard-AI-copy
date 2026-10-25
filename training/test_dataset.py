
import os
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from detectors.xception_detector import XceptionDetector

# Run from the training directory
DATASET_DIR = os.path.abspath("../dataset/external/archive/Human Faces Dataset")
CHECKPOINT = os.path.abspath("../xception.pth")

REAL_DIR = os.path.join(DATASET_DIR, "Real Images")
FAKE_DIR = os.path.join(DATASET_DIR, "AI-Generated Images")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Dataset:", DATASET_DIR)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def get_images(folder, label):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    paths = []
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                paths.append((os.path.join(root, filename), label))

    print(f"{'Real' if label == 0 else 'Fake'} images found: {len(paths)}")
    return paths


def get_logits(model, image):
    # The repository's detector normally returns a dictionary.
    output = model({"image": image})

    if isinstance(output, dict):
        if "cls" not in output:
            raise RuntimeError(
                f"Model output dictionary has no 'cls' key. Keys: {list(output.keys())}"
            )
        logits = output["cls"]
    elif torch.is_tensor(output):
        logits = output
    else:
        raise TypeError(f"Unexpected model output type: {type(output)}")

    # Expected shapes include [1, 1] for binary classification
    # or [1, 2] for two-class classification.
    if logits.ndim == 0:
        logits = logits.reshape(1, 1)
    elif logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if logits.ndim != 2 or logits.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected classifier output shape: {tuple(logits.shape)}"
        )

    if logits.shape[1] == 1:
        # Single-logit binary classifier: positive logit means fake.
        fake_probability = torch.sigmoid(logits[0, 0]).item()
    elif logits.shape[1] == 2:
        # Two-class classifier: class 0 = real, class 1 = fake.
        fake_probability = torch.softmax(logits, dim=1)[0, 1].item()
    else:
        raise RuntimeError(
            f"Expected 1 or 2 classifier outputs, got {logits.shape[1]}"
        )

    return fake_probability


def main():
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    samples = get_images(REAL_DIR, 0) + get_images(FAKE_DIR, 1)

    if not samples:
        raise RuntimeError("No images found in the dataset folders.")

    model = XceptionDetector().to(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    y_true = []
    y_pred = []
    skipped = 0

    with torch.inference_mode():
        for index, (path, label) in enumerate(samples, start=1):
            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                    tensor = transform(image).unsqueeze(0).to(device)

                fake_probability = get_logits(model, tensor)
                prediction = int(fake_probability >= 0.5)

                y_true.append(label)
                y_pred.append(prediction)

            except Exception as error:
                skipped += 1
                # Stop immediately for model/output errors instead of
                # printing thousands of identical messages.
                if isinstance(error, (RuntimeError, TypeError)):
                    raise RuntimeError(
                        f"Evaluation failed on {path}: {error}"
                    ) from error

                print(f"Skipped unreadable image {path}: {error}")

            if index % 100 == 0 or index == len(samples):
                print(f"Processed {index}/{len(samples)} images")

    if not y_true:
        raise RuntimeError("No images were successfully evaluated.")

    print("\n========== EXTERNAL DATASET RESULTS ==========")
    print("Images evaluated:", len(y_true))
    print("Images skipped:", skipped)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(
        f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}"
    )
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print("Order: Real (0), Fake (1)")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["Real", "Fake"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
