# OCR Text Translator 🌍

A computer vision application that extracts text from images or videos and translates it into different languages. This project handles various text fonts, sizes, widths, and orientations, making it perfect for travelers or anyone working with multilingual content.

## Features

- ✨ **Multi-language OCR**: Detects text in multiple languages simultaneously
- 🌐 **Automatic Translation**: Translates extracted text to 40+ languages
- 🎯 **Robust Detection**: Handles various fonts, sizes, and orientations
- 🖼️ **Image Preprocessing**: Enhances image quality for better OCR accuracy
- 📹 **Video Support**: Process video frames for text extraction
- 🎨 **Visualization**: Draws bounding boxes around detected text
- 🌐 **Web Interface**: User-friendly Streamlit web application

## Technology Stack

- **Python 3.8+**: Primary programming language
- **EasyOCR**: State-of-the-art OCR engine supporting 80+ languages
- **Helsinki-NLP Transformer Models**: Local, offline translation models (no API required)
- **PyTorch**: Deep learning framework for translation models
- **OpenCV**: Image processing and preprocessing
- **Streamlit**: Web interface framework

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone or Download the Project

```bash
cd R:\project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: 
- On first run, EasyOCR will download language models automatically. This may take a few minutes.
- Translation models will be downloaded automatically on first use (one-time download per language pair).
- All processing is done **locally** - no internet connection required after initial setup (except for downloading models).

### Step 3: Verify Installation

```bash
python -c "import easyocr; import cv2; print('Installation successful!')"
```

## Usage

### Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and:
1. Upload an image containing text
2. Select OCR detection languages
3. Choose target translation language
4. Click "Extract & Translate"

### Command Line Interface

Basic usage:

```bash
python main.py path/to/image.jpg --target-lang es
```

With options:

```bash
python main.py image.jpg \
    --target-lang fr \
    --ocr-langs en es fr \
    --draw-boxes \
    --output annotated_image.jpg \
    --json results.json
```

#### Command Line Arguments

- `image_path`: Path to input image (required)
- `--target-lang`: Target language code (default: 'en')
- `--ocr-langs`: Space-separated OCR detection languages (default: ['en'])
- `--no-preprocess`: Skip image preprocessing
- `--draw-boxes`: Draw bounding boxes on detected text
- `--output`: Save annotated image to this path
- `--json`: Save results to JSON file

### Python API

```python
from main import ImageTextTranslator

# Initialize translator
translator = ImageTextTranslator(ocr_languages=['en', 'es'])

# Process image
results = translator.process_image(
    'image.jpg',
    target_language='fr',
    preprocess=True,
    draw_boxes=True
)

# Access results
for original, translation in zip(
    results['detected_texts'],
    results['translations']
):
    print(f"Original: {original}")
    print(f"Translated: {translation['translated_text']}")
```

## Supported Languages

### OCR Detection (80+ languages)
English, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic, Russian, Hindi, and many more.

### Translation (30+ languages via Helsinki-NLP models)
English, Spanish, French, German, Italian, Portuguese, Russian, Chinese, Arabic, Hindi, Dutch, Polish, Turkish, Vietnamese, Thai, Czech, Swedish, Danish, Norwegian, Finnish, Greek, Hebrew, Indonesian, Malay, Ukrainian, Romanian, Hungarian, Slovak, Bulgarian, Croatian, Serbian, Slovenian, Estonian, Latvian, Lithuanian, and more.

**Note**: Translation models are downloaded automatically on first use. Each language pair requires a separate model download (one-time).

## Project Structure

```
project/
├── ocr_translator.py    # OCR text extraction module
├── translator.py         # Translation module
├── main.py              # Main application and CLI
├── app.py               # Streamlit web interface
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## How It Works

1. **Image Preprocessing** (optional):
   - Converts to grayscale
   - Applies denoising
   - Enhances contrast using CLAHE
   - Applies thresholding

2. **Text Detection**:
   - Uses EasyOCR to detect text regions
   - Handles various fonts, sizes, and orientations
   - Returns text with bounding box coordinates

3. **Text Translation**:
   - Uses detected language from OCR as source language
   - Translates to target language using local transformer models (Helsinki-NLP)
   - Works completely offline after initial model download
   - Returns original and translated text

4. **Visualization** (optional):
   - Draws bounding boxes around detected text
   - Labels text regions

## Examples

### Example 1: Translate Spanish Menu
```bash
python main.py spanish_menu.jpg --target-lang en --ocr-langs es en
```

### Example 2: Translate Japanese Sign
```bash
python main.py japanese_sign.jpg --target-lang en --ocr-langs ja en --draw-boxes
```

### Example 3: Process Multiple Languages
```bash
python main.py multilingual_image.jpg --target-lang en --ocr-langs en es fr de
```

## Troubleshooting

### Issue: "No text detected"
- Try selecting different OCR languages
- Enable preprocessing (default: enabled)
- Ensure image has good contrast
- Use higher resolution images

### Issue: Slow processing
- First run downloads language models (one-time)
- Processing time depends on image size and complexity
- Consider reducing OCR languages if speed is critical

### Issue: Translation errors
- First-time model downloads require internet connection
- Some language pairs may not be available (check Helsinki-NLP model availability)
- Ensure sufficient disk space for model storage (~500MB per model)
- Try different target language codes

## Performance Tips

1. **Select specific OCR languages**: Only include languages you expect in the image
2. **Image quality**: Higher resolution images yield better results
3. **Preprocessing**: Usually improves accuracy but adds processing time
4. **Batch processing**: Process multiple images programmatically using the API

## Future Enhancements

- [ ] Support for handwritten text
- [ ] Real-time video processing
- [ ] Batch image processing
- [ ] Custom OCR model training
- [ ] Integration with cloud storage
- [ ] Better language detection per text segment

## License

This project is open source and available for educational and personal use.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- EasyOCR: https://github.com/JaidedAI/EasyOCR
- Helsinki-NLP Translation Models: https://huggingface.co/Helsinki-NLP
- Transformers Library: https://huggingface.co/docs/transformers/
- OpenCV: https://opencv.org/
- Streamlit: https://streamlit.io/

---

**Made with ❤️ for travelers and multilingual content creators**
