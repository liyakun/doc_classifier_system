import pytesseract
from PIL import Image

class OCR:
    def __init__(self):
        pass

    @staticmethod
    def ocr_doc(image_f):
        """
        ocr from input image
        """
        text = pytesseract.image_to_string(Image.open(image_f).convert('L'))
        return text
