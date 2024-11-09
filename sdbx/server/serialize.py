import io
import json
import base64

from PIL import Image

class WebEncoder(json.JSONEncoder):
    @staticmethod
    def serialize_image(image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")  # Save the image in PNG format
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

    def default(self, obj):
        if isinstance(obj, Image.Image):
            return WebEncoder.serialize_image(obj)
        try:
            return super().default(obj)
        except (TypeError, OverflowError, ValueError):
            return str(data)