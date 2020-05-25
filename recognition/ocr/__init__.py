try:
    from recognition.ocr import *
    from recognition.ocr.utils import *
    from recognition.ocr.evaluate_cer import *
    from recognition.ocr.handwriting_line_recognition import *
    from recognition.ocr.paragraph_segmentation_dcnn import *
    from recognition.ocr.word_and_line_segmentation import *
except:
    try:
        from handwritten_text_recognition.recognition.ocr import *
        from handwritten_text_recognition.recognition.ocr.utils import *
        from handwritten_text_recognition.recognition.ocr.evaluate_cer import *
        from handwritten_text_recognition.recognition.ocr.handwriting_line_recognition import *
        from handwritten_text_recognition.recognition.ocr.paragraph_segmentation_dcnn import *
        from handwritten_text_recognition.recognition.ocr.word_and_line_segmentation import *
    except:
        from recognition.handwritten_text_recognition.recognition.ocr import *
        from recognition.handwritten_text_recognition.recognition.ocr.utils import *
        from recognition.handwritten_text_recognition.recognition.ocr.evaluate_cer import *
        from recognition.handwritten_text_recognition.recognition.ocr.handwriting_line_recognition import *
        from recognition.handwritten_text_recognition.recognition.ocr.paragraph_segmentation_dcnn import *
        from recognition.handwritten_text_recognition.recognition.ocr.word_and_line_segmentation import *
