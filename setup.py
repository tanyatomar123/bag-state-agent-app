# setup.py
from setuptools import setup, find_packages

setup(
    name="bag-detection-agent",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.28.0",
        "opencv-python==4.8.1.78",
        "numpy==1.24.3",
        "paddleocr==2.7.0",
        "easyocr==1.7.0",
        "pytesseract==0.3.10",
        "Pillow==10.0.1",
        "torch==2.0.1",
        "torchvision==0.15.2",
        "python-multipart==0.0.6",
        "plotly==5.15.0",
        "pandas==2.0.3",
        "altair==4.2.2",
    ],
    python_requires=">=3.8",
)
