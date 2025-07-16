# 🛂 Passport OCR Scanner

This project is a web-based **Passport OCR Scanner** tool built using **Python**, **Streamlit**, and **Tesseract OCR**, designed to extract Machine Readable Zone (MRZ) data and Date of Birth from passport images. It applies advanced image preprocessing and custom region detection techniques to ensure high accuracy in real-world passport scans.

---

## 📸 Features

- 🧠 **MRZ Parsing**: Detects and decodes 2-line MRZ codes from passports (TD3 format).
- 📅 **DOB Extraction**: Accurately extracts and parses the date of birth (manual or auto region-based).
- 🎛 **Image Preprocessing Options**:
  - Grayscale conversion
  - Contrast and brightness adjustments
  - Denoising (median blur, fast N-means)
  - Thresholding (Otsu, adaptive mean/gaussian)
  - Sharpening, contrast enhancement (CLAHE)
  - Deskewing and auto region refinement
- 🎯 Manual region adjustment for non-standard documents.
- 📥 Downloadable CSV of extracted fields.
- 💻 Web-based UI using **Streamlit**.
- 🧪 Fully customizable parameters in the sidebar.

---

## 🧰 Technologies Used

| Component     | Description                       |
|---------------|-----------------------------------|
| Python 3.x     | Core language                    |
| Streamlit      | Web interface framework          |
| OpenCV         | Image processing and enhancement |
| Tesseract OCR  | Optical character recognition    |
| PIL (Pillow)   | Image handling                   |
| NumPy          | Numerical processing             |
| Regex, CSV     | Text parsing and export          |

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/passport-ocr-scanner.git
cd passport-ocr-scanner
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Make sure `tesseract` is installed and accessible via your system PATH or defined in `main_clearphotos.py`.

* **Windows Installer**: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)

### 3. Run the App

```bash
streamlit run main_clearphotos.py
```

Open the link (usually [http://localhost:8501](http://localhost:8501)) in your browser to interact with the tool.

---

## 📁 Folder Structure

```bash
passport-ocr-scanner/
├── main_clearphotos.py        # Main Streamlit app
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── sample_passports/          # (optional) Test images
```

---

## 🖼 Sample Output

* MRZ: `P<MYSLEE<<KHINE<<<<<<<<<<<<<<<<<<<<<<<<<<<`
* Passport No: `AC1234567`
* DOB: `07 JAN 1995`
* Nationality: `MMR`

---

## 📝 Output Format

Data is summarized on screen and downloadable as a `.csv` file with fields like:

* Date of Birth
* Document Type
* Code (Country)
* Surname
* Given Names
* Passport Number
* MRZ Raw
* Parse Status

---

## 📦 Requirements

* Python 3.7+
* Tesseract OCR (v4+)
* Libraries from `requirements.txt`

---

## 🛠 Future Improvements

* MRZ signature verification
* OCR confidence score visualization
* Multi-language and multi-document support (NRC, NID, etc.)
* Layout detection using AI (e.g., LayoutLM or Donut)

---
