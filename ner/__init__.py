# from .do_ner import NER_MODEL

# ner_model = NER_MODEL()
import os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
filePath = os.path.split(curPath)[0]
sys.path.append(filePath)
sys.path.extend([filePath + '\\' + i for i in os.listdir(filePath)])