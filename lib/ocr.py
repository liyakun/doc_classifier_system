import re
import isbnlib
import datefinder
import pytesseract
from PIL import Image


class OCR:

    @staticmethod
    def ocr_doc(image_f):
        """
        ocr from input image
        :param image_f read content from image
        :return return text content, and date
        """
        text = pytesseract.image_to_string(Image.fromarray(image_f).convert('L'))
        return text

    @staticmethod
    def parse_date(text):
        """
        find dates from input text
        :param text:
        :return: dates
        """
        return [datet.isoformat() for datet in list(datefinder.find_dates(str(text)))]

    @staticmethod
    def parse_isbn(text):
        """
        parse isbn number from input text
        reference:
        :param text:
        :return: isbn number
        """
        return isbnlib.get_isbnlike(str(text), level='normal')

    @staticmethod
    def date_isbn_extraction(classification_result, ocr_text):
        """
        extract date or isbn from text according to classification result
        :param classification_result:
        :param ocr_text:
        :return: result type(date or isbn), and value
        """
        # define the document type to extract date
        date_classes = ['email', 'form', 'letter', 'news']
        # get the first ranked classification type
        doc_class = classification_result[0].split('-')[1].replace('"', '').strip()
        print(doc_class)
        extraction_type = 'date'
        if doc_class in date_classes:
            extraction_result = OCR.parse_date(text=ocr_text)
        else:
            extraction_type = 'isbn'
            extraction_result = OCR.parse_isbn(text=ocr_text)
        return extraction_type, extraction_result
