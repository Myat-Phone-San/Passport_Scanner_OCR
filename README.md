Passport Optical Character Recognition (OCR) System
Overview
This project implements a Passport Optical Character Recognition (OCR) system designed to extract key data from passport documents using computer vision and machine learning techniques. Developed for the Ministry of Hotels and Tourism, this system automates the process of digitizing information such as date of birth, passport number, and other relevant details from passport images.

The system features a user-friendly web interface built with Streamlit, allowing easy image uploads and real-time display of preprocessing effects and extracted data.

Features
Automated Data Extraction: Automatically identifies and extracts specific data fields (e.g., passport number, name, date of birth, expiry date) from scanned or photographed passport images.

Image Pre-processing: Includes steps to enhance image quality for better OCR accuracy, such as de-skewing, noise reduction, and contrast adjustment.

Robust OCR Engine: Leverages powerful OCR capabilities to convert visual text into machine-readable text.

Structured Output: Organizes extracted data into a structured format for easy integration with databases or other systems.

Interactive Streamlit UI: Provides a web-based interface for uploading images, adjusting preprocessing parameters, and visualizing the results.

Technologies Used
Python: The primary programming language for development.

Streamlit: For creating the interactive web application interface.

OpenCV (Open Source Computer Vision Library): Used for image processing tasks, including image loading, manipulation, and preparing images for OCR.

Tesseract OCR: An open-source OCR engine used for recognizing text in images.

Pillow (PIL Fork): For image manipulation in Python, often used in conjunction with Streamlit.

NumPy: For numerical operations, especially with image data.

pytesseract: Python wrapper for Tesseract OCR.

Image Processing Libraries/Techniques: Various algorithms and methods applied to optimize image quality for accurate text recognition.

Installation
To set up and run this project locally, follow these steps:

Prerequisites
Python 3.x

pip (Python package installer)

Tesseract OCR engine installed on your system.

For Ubuntu/Debian: sudo apt-get install tesseract-ocr

For macOS (using Homebrew): brew install tesseract

For Windows: Download the installer from the Tesseract GitHub page and ensure it's added to your system's PATH.

Steps
Clone the repository:

git clone https://github.com/Myat-Phone-San/Passport-OCR-System.git
cd Passport-OCR-System

(Note: Replace https://github.com/Myat-Phone-San/Passport-OCR-System.git with the actual GitHub repository URL for this specific project if it's different from your main profile.)

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install the required Python packages:

pip install -r requirements.txt

Example requirements.txt content:

streamlit
opencv-python
pytesseract
Pillow
numpy

Usage
This project is implemented as a Streamlit web application.

Save the application code:
Save the provided Python code (the large block you shared) into a file named app.py (or any other .py file) within the cloned project directory.

Run the Streamlit application:
Open your terminal or command prompt, navigate to the project directory (where app.py is located), and run the following command:

streamlit run main_clearphotos.py

This command will open the Streamlit application in your web browser. You can then upload passport images, adjust preprocessing settings in the sidebar, and view the extracted MRZ and Date of Birth information.

Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

Fork the repository.

Create a new branch (git checkout -b feature/YourFeature).

Make your changes.

Commit your changes (git commit -m 'Add new feature').

Push to the branch (git push origin feature/YourFeature).

Open a Pull Request.
