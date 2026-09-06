from io import BytesIO
from urllib import response
from fastapi.testclient import TestClient
from PIL import Image
from src.api import app, get_predictor


from src.api import app

class FakePredictor:
    def predict(self, image_path, indication):
        return {
            "image_probability": 0.2,
            "text_probability": 0.4,
            "fusion_probability": 0.24,
            "fusion_alpha": 0.8,
        }


client = TestClient(app)

def create_test_image():
    image = Image.new(
        "RGB",
        (224, 224),
        color="black",
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input():
    image_file = create_test_image()

    response = client.post(
        "/predict",
        files={
            "image": (
                "test.png",
                image_file,
                "image/png",
            )
        },
        data={
            "indication": "Shortness of breath and chest pain"
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "image_probability" in result
    assert "text_probability" in result
    assert "fusion_probability" in result
    assert "fusion_alpha" in result

    assert 0 <= result["image_probability"] <= 1
    assert 0 <= result["text_probability"] <= 1
    assert 0 <= result["fusion_probability"] <= 1
    assert result["fusion_alpha"] == 0.8


def test_predict_empty_indication():
    image_file = create_test_image()

    response = client.post(
            "/predict",
            files={
                "image": (
                    "test.png",
                    image_file,
                    "image/png",
                )
            },
            data={
                "indication": "   "
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Indication cannot be empty."


def test_predict_invalid_file_type():
        response = client.post(
        "/predict",
        files={
            "image": (
                "not_an_image.txt",
                b"hello",
                "text/plain",
            )
        },
        data={
            "indication": "Shortness of breath"
        },
    )

        assert response.status_code == 415
        assert (response.json()["detail"]
        == "Image must be a PNG or JPEG file."
    )

app.dependency_overrides[get_predictor] = lambda: FakePredictor()

def test_predict_corrupted_image():
        response = client.post(
        "/predict",
        files={
            "image": (
                "fake.png",
                b"this is not actually a png",
                "image/png",
            )
        },
        data={
            "indication": "Shortness of breath"
        },
    )

        assert response.status_code == 400
        assert response.json()["detail"] == "Uploaded file is not a valid image."