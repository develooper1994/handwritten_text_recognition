try:
    import recognition.recognizer
    import recognition.get_models
    import recognition.ocr
    import recognition.tests
    import recognition.utils
except:
    try:
        import handwritten_text_recognition.recognition.recognizer
        import handwritten_text_recognition.recognition.get_models
        import handwritten_text_recognition.recognition.ocr
        import handwritten_text_recognition.recognition.tests
        import handwritten_text_recognition.recognition.utils
    except:
        import recognition.handwritten_text_recognition.recognition.recognizer
        import recognition.handwritten_text_recognition.recognition.get_models
        import recognition.handwritten_text_recognition.recognition.ocr
        import recognition.handwritten_text_recognition.recognition.tests
        import recognition.handwritten_text_recognition.recognition.utils