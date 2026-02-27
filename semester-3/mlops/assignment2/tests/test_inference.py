import numpy as np
from src.inference import preprocess

def test_preprocess_shape():
    # fake image bytes simulation
    import io
    from PIL import Image

    img = Image.new("RGB", (300,300))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    arr = preprocess(buf.getvalue())

    assert arr.shape == (1,224,224,3)