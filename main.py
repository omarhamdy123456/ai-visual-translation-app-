"""
Main application for OCR Text Translation.
Combines OCR and translation functionality.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()
from ocr_translator import OCRTranslator
from translator import TextTranslator
from typing import List, Tuple, Optional
import json


class ImageTextTranslator:
    """Complete pipeline for extracting and translating text from images."""
    
    def __init__(
        self,
        ocr_languages: List[str] = ['en'],
        ocr_backend: str = 'easyocr',
        trocr_model: Optional[str] = None,
    ):
        """
        Initialize the translator pipeline.
        
        Args:
            ocr_languages: Languages for OCR detection (e.g., ['en', 'es', 'fr'])
            ocr_backend: easyocr, paddleocr, or trocr_hybrid (EasyOCR detect + TrOCR read)
            trocr_model: Optional Hugging Face id for TrOCR when using trocr_hybrid
        """
        ocr_kwargs: dict = {'languages': ocr_languages, 'backend': ocr_backend}
        if trocr_model and (ocr_backend or '').strip().lower() == 'trocr_hybrid':
            ocr_kwargs['trocr_model'] = trocr_model
        self.ocr = OCRTranslator(**ocr_kwargs)
        self.translator = TextTranslator()
    
    def process_image(
        self,
        image_path: str,
        target_language: str = 'en',
        preprocess: bool = True,
        draw_boxes: bool = False,
        output_image_path: Optional[str] = None,
        max_dim: int = 1200,
        small_text_boost: bool = False,
        high_recall: bool = False,
    ) -> dict:
        """
        Complete pipeline: Extract text from image and translate it.
        
        Args:
            image_path: Path to input image
            target_language: Target language code for translation
            preprocess: Whether to preprocess image before OCR
            draw_boxes: Whether to draw bounding boxes on detected text
            output_image_path: Optional path to save annotated image
            max_dim: Max image dimension for OCR processing
            small_text_boost: Boost OCR for smaller/thinner text
            high_recall: Lower thresholds to catch more text (may add noise)
            
        Returns:
            Dictionary with extraction and translation results
        """
        # Extract text
        print(f"Extracting text from {image_path}...")
        ocr_results = self.ocr.extract_text(
            image_path,
            preprocess=preprocess,
            max_dim=max_dim,
            small_text_boost=small_text_boost,
            high_recall=high_recall,
        )
        
        if not ocr_results:
            return {
                'image_path': image_path,
                'detected_texts': [],
                'translations': [],
                'message': 'No text detected in the image'
            }
        
        # Extract text strings and detected languages
        detected_texts = []
        detected_languages = []
        for text, _, lang, *_ in ocr_results:
            detected_texts.append(text)
            detected_languages.append(lang)
        
        # Use the most common detected language as source language
        # If all texts are in the same language, use that; otherwise use the first OCR language
        source_language = detected_languages[0] if detected_languages else 'en'
        if len(set(detected_languages)) == 1:
            source_language = detected_languages[0]
        
        # Translate texts
        print(f"Translating {len(detected_texts)} text segments from {source_language} to {target_language}...")
        translations = self.translator.translate_batch(detected_texts, target_language, source_language)
        
        # Draw bounding boxes if requested
        annotated_image = None
        if draw_boxes:
            annotated_image = self.ocr.draw_bounding_boxes(
                image_path, ocr_results, output_image_path
            )
        
        return {
            'image_path': image_path,
            'detected_texts': detected_texts,
            'ocr_results': ocr_results,
            'translations': translations,
            'annotated_image': annotated_image is not None
        }
    
    def process_video(self, video_path: str, target_language: str = 'en',
                     frame_interval: int = 30, output_dir: Optional[str] = None) -> List[dict]:
        """
        Process video frames for text extraction and translation.
        
        Args:
            video_path: Path to input video
            target_language: Target language code for translation
            frame_interval: Process every Nth frame
            output_dir: Optional directory to save frame results
            
        Returns:
            List of results for each processed frame
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        results = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                print(f"Processing frame {frame_count}...")
                ocr_results = self.ocr.extract_text_from_video_frame(frame)
                
                if ocr_results:
                    detected_texts = []
                    detected_languages = []
                    for text, _, lang, *_ in ocr_results:
                        detected_texts.append(text)
                        detected_languages.append(lang)
                    
                    source_language = detected_languages[0] if detected_languages else 'en'
                    translations = self.translator.translate_batch(detected_texts, target_language, source_language)
                    
                    result = {
                        'frame_number': frame_count,
                        'detected_texts': detected_texts,
                        'translations': translations
                    }
                    results.append(result)
                    
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                        cv2.imwrite(frame_path, frame)
            
            frame_count += 1
        
        cap.release()
        return results


def main():
    """Command-line interface for the translator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='OCR Text Translator')
    parser.add_argument('image_path', help='Path to input image')
    parser.add_argument('--target-lang', default='en', help='Target language code (default: en)')
    parser.add_argument('--ocr-langs', nargs='+', default=['en'], 
                       help='OCR detection languages (default: en)')
    parser.add_argument(
        '--ocr-backend',
        default='easyocr',
        choices=['easyocr', 'paddleocr', 'trocr_hybrid'],
        help='OCR engine: easyocr, paddleocr, or trocr_hybrid (handwriting; EasyOCR+TrOCR)',
    )
    parser.add_argument(
        '--trocr-model',
        default='',
        help='Hugging Face TrOCR model id (trocr_hybrid). Default: env OCR_TROCR_MODEL or base-handwritten',
    )
    parser.add_argument('--no-preprocess', action='store_true', 
                       help='Skip image preprocessing')
    parser.add_argument('--draw-boxes', action='store_true', 
                       help='Draw bounding boxes on detected text')
    parser.add_argument(
        '--ocr-max-dim',
        type=int,
        default=1200,
        help='Max image dimension for OCR (higher helps handwriting, slower)',
    )
    parser.add_argument(
        '--small-text-boost',
        action='store_true',
        help='Use stronger OCR upscaling for thin/fine text',
    )
    parser.add_argument(
        '--high-recall',
        action='store_true',
        help='Lower OCR thresholds to detect more text (can add noise)',
    )
    parser.add_argument(
        '--handwriting-mode',
        action='store_true',
        help='Preset for handwriting: enables preprocessing, high recall, and stronger scaling',
    )
    parser.add_argument('--output', help='Output path for annotated image')
    parser.add_argument('--json', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    preprocess = not args.no_preprocess
    ocr_max_dim = args.ocr_max_dim
    small_text_boost = args.small_text_boost
    high_recall = args.high_recall

    if args.handwriting_mode:
        # Handwriting usually needs extra recall + higher working resolution.
        preprocess = True
        small_text_boost = True
        high_recall = True
        ocr_max_dim = max(ocr_max_dim, 2200)

    # Initialize translator
    translator = ImageTextTranslator(
        ocr_languages=args.ocr_langs,
        ocr_backend=args.ocr_backend,
        trocr_model=(args.trocr_model or '').strip() or None,
    )
    
    # Process image
    results = translator.process_image(
        args.image_path,
        target_language=args.target_lang,
        preprocess=preprocess,
        draw_boxes=args.draw_boxes,
        output_image_path=args.output,
        max_dim=ocr_max_dim,
        small_text_boost=small_text_boost,
        high_recall=high_recall,
    )
    
    # Print results
    print("\n" + "="*60)
    print("OCR TEXT TRANSLATION RESULTS")
    print("="*60)
    
    if results['detected_texts']:
        print(f"\nDetected {len(results['detected_texts'])} text segment(s):\n")
        for i, (original, translation) in enumerate(
            zip(results['detected_texts'], results['translations']), 1
        ):
            print(f"Segment {i}:")
            print(f"  Original: {original}")
            print(f"  Translated ({translation['target_language']}): {translation['translated_text']}")
            print()
    else:
        print("\nNo text detected in the image.")
    
    # Save to JSON if requested
    if args.json:
        # Remove non-serializable items
        json_results = {
            'image_path': results['image_path'],
            'detected_texts': results['detected_texts'],
            'translations': [
                {
                    'original_text': t['original_text'],
                    'translated_text': t['translated_text'],
                    'source_language': t['source_language'],
                    'target_language': t['target_language']
                }
                for t in results['translations']
            ]
        }
        with open(args.json, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.json}")


if __name__ == '__main__':
    import cv2
    main()
