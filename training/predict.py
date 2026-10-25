# import argparse
# import torch
# from PIL import Image
# from torchvision import transforms
# from detectors.xception_detector import XceptionDetector


# def main():
#     parser = argparse.ArgumentParser(description="FaceGuard AI Image Detector")
#     parser.add_argument("image", help="Path to the image")
#     parser.add_argument(
#         "--checkpoint",
#         default="../xception.pth",
#         help="Path to the trained model checkpoint",
#     )
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Device: {device}")

#     model = XceptionDetector()

#     checkpoint = torch.load(
#         args.checkpoint,
#         map_location="cpu",
#         weights_only=True
#     )

#     # Support either a raw state_dict or a wrapped checkpoint.
#     if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
#         checkpoint = checkpoint["state_dict"]
#     elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#         checkpoint = checkpoint["model_state_dict"]

#     model.load_state_dict(checkpoint)
#     model.to(device)
#     model.eval()

#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             [0.5, 0.5, 0.5],
#             [0.5, 0.5, 0.5]
#         ),
#     ])

#     with Image.open(args.image) as image:
#         image = image.convert("RGB")
#         tensor = transform(image).unsqueeze(0).to(device)

#     with torch.inference_mode():
#         result = model({"image": tensor}, inference=True)
#         logit = result["cls"].reshape(-1)[0]
#         fake_probability = torch.sigmoid(logit).item()

#     real_probability = 1.0 - fake_probability
#     prediction = (
#         "AI-GENERATED / FAKE"
#         if fake_probability >= 0.5
#         else "REAL"
#     )

#     print("\n========== FACEGUARD PREDICTION ==========")
#     print(f"Image: {args.image}")
#     print(f"Raw logit: {logit.item():.6f}")
#     print(f"Fake probability: {fake_probability:.2%}")
#     print(f"Real probability: {real_probability:.2%}")
#     print(f"Prediction: {prediction}")
#     print("==========================================")

#     print(
#         "Note: Probabilities are model scores, not guaranteed "
#         "real-world confidence estimates."
#     )


# if __name__ == "__main__":
#     main()
import argparse
import torch
from PIL import Image
from torchvision import transforms
from detectors.xception_detector import XceptionDetector


def main():
    parser = argparse.ArgumentParser(description="FaceGuard AI Image Detector")
    parser.add_argument("image", help="Path to the image")
    parser.add_argument(
        "--checkpoint",
        default="../xception.pth",
        help="Path to the trained model checkpoint",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = XceptionDetector()

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True
    )

    # Support either a raw state_dict or a wrapped checkpoint.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ])

    with Image.open(args.image) as image:
        image = image.convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        result = model({"image": tensor}, inference=True)
        logit = result["cls"].reshape(-1)[0]
        fake_probability = torch.sigmoid(logit).item()

    real_probability = 1.0 - fake_probability
    prediction = (
        "AI-GENERATED / FAKE"
        if fake_probability >= 0.5
        else "REAL"
    )

    print("\n========== FACEGUARD PREDICTION ==========")
    print(f"Image: {args.image}")
    print(f"Raw logit: {logit.item():.6f}")
    print(f"Fake probability: {fake_probability:.2%}")
    print(f"Real probability: {real_probability:.2%}")
    print(f"Prediction: {prediction}")
    print("==========================================")

    print(
        "Note: Probabilities are model scores, not guaranteed "
        "real-world confidence estimates."
    )


if __name__ == "__main__":
    main()