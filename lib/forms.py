from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired


class ImageForm(FlaskForm):
	"""
	An image form for uploading
	"""
	image = FileField('image', validators=[FileRequired(message="Please include 'image' field.")])
