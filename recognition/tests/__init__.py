try:
    from recognition.tests.tests import *
except:
    try:
        from handwritten_text_recognition.recognition.tests.tests import *
    except:
        from recognition.handwritten_text_recognition.recognition.tests.tests import *
