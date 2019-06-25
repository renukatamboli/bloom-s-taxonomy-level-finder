from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from rest_framework.renderers import TemplateHTMLRenderer
from .models import File
#from .views import FileView
from django.http import HttpResponse
from .serializers import FileSerializer
from string import digits
from subprocess import call
#pdf
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
#from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
#from pdfminer.converter import TextConverter
#from pdfminer.layout import LAParams
#from pdfminer.pdfpage import PDFPage
#from cStringIO import StringIO
#image
#from PIL import Image
import pytesseract
import cv2
import re
import docx
import docx2txt
import os
#ML
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import ngrams
from nltk.corpus import stopwords
import nltk
import random
#from textblob.classifiers import NaiveBayesClassifier
#from textblob import TextBlob
from wand.image import Image
from PIL import Image as PI
import pyocr
import pyocr.builders
import io
import codecs
import uuid
import numpy as np
import glob
import os
import sys
class FileView(APIView):

  parser_classes = (MultiPartParser, FormParser)

  renderer_classes = [TemplateHTMLRenderer]
  template_name = 'file.html'
  def post(self, request, *args, **kwargs):

    file_serializer = FileSerializer(data=request.data)
    if file_serializer.is_valid():
      file_serializer.save()
      file_path = "/home/bloomtaxo/fileupload"+file_serializer.data["file"]
      q = open('/home/bloomtaxo/fileupload/qextract/question.txt').read().lower()
      i = open('/home/bloomtaxo/fileupload/qextract/notquestion.txt').read().lower()
      bt11 = open('/home/bloomtaxo/fileupload/bt/bt1.txt').read().lower()
      bt22 = open('/home/bloomtaxo/fileupload/bt/bt2.txt').read().lower()
      bt33 = open('/home/bloomtaxo/fileupload/bt/bt3.txt').read().lower()
      bt44 = open('/home/bloomtaxo/fileupload/bt/bt4.txt').read().lower()
      bt55 = open('/home/bloomtaxo/fileupload/bt/bt5.txt').read().lower()
      bt66 = open('/home/bloomtaxo/fileupload/bt/bt6.txt').read().lower()
      bt1=['define','describe','draw','find','identify','label','list','locate','match','memorise','name','recall','recite','recognize','relate','reproduce','select','state','tell','write']
      bt2=['compare','convert','demonstarte','describe','discuss','distinguish','explain','find out more information about','generalize','interpret','outline','paraphrase','predict','put into your own words','relate','restate','summarize','translate','visualize']
      bt3=['apply','calculate','change','choose','complete','construct','examine','illustrate','interpret','make','manipulate','modify','produce','put into practice','put together','solve','show','translate','use']
      bt4 = ['advertise','analyse','categoriase','compare','contrast','deduce','differenciate','distinguish','examine','explain','identify','investigate','seperate','subdivide','take apart']
      bt5 = ['argue','assess','choose','compose','construct','create','criticise','critique','debate','decide','defend','design','determine','device','discuss','estimate','evaluate','formulate','imagine','invent','judge','justify','plan','predict','prioritise','propose','rate','recommend','select','value']
      bt6 = ['add to','argue','assess','choose','combine','compose','construct','create','debate','decide','design','determine','devise','discuss','forcast','formulate','hypothesise','imagine','invent','judge','justify','originate','plan','predict','priortise','propose','rate','recommend','select','verify']
      bt = {'bt1': bt1, 'bt2': bt2,'bt3':bt3,'bt4':bt4,'bt5':bt5,'bt6':bt6}
      btperq = []
      #stop_words = set(stopwords.words('english'))
      def get_features(sentence):
          word_tokens = word_tokenize(sentence)
          #filtered_sentence = [w for w in word_tokens if not w in stop_words]
          return {"first_word":word_tokens[0],"last_word":word_tokens[-1]}
      labeled_sentence = ([(name, "question") for name in sent_tokenize(q)] +[(name, "notquestion") for name in sent_tokenize(i)])
      random.shuffle(labeled_sentence)
      feature_sets = [(get_features(n), label) for (n, label) in labeled_sentence]
      #train_set, test_set = feature_sets[500:], feature_sets[:500]
      classifier = nltk.NaiveBayesClassifier.train(feature_sets)
      #bt classification
      def get_btfeatures(sentence):
          word_tokens = word_tokenize(sentence)
          c=nltk.pos_tag(["we"]+word_tokens)[1:]
          words,tags = zip(*c)
          stop = ["$","''","(",")",",",":","CC","DT","CD","EX","FW","NN","NNP","NNS","PDT","PRP","PRP$","SYM","``","TO","."]
          stems = [words for words,tags in c if not tags in stop]
          gram =[]
          grams = ngrams(list(stems), len(stems))
          for g in grams:
              gram = g
          if(gram == []):
              return{"stem":tuple(word_tokens)}
          else:
              return{"stem":gram}

      labeled_sentencebt = ([(name, "bt1") for name in sent_tokenize(bt11)] +[(name, "bt2") for name in sent_tokenize(bt22)]+[(name, "bt3") for name in sent_tokenize(bt33)]+[(name, "bt4") for name in sent_tokenize(bt44)]+[(name, "bt5") for name in sent_tokenize(bt55)]+[(name, "bt6") for name in sent_tokenize(bt66)])
      random.shuffle(labeled_sentencebt)
      feature_setsbt = [(get_btfeatures(n), label) for (n, label) in labeled_sentencebt]
      btclassifier = nltk.NaiveBayesClassifier.train(feature_setsbt)
      t = file_path.split(".")
      if(t[1]=="pdf"):
          """def convert_pdf_to_txt(path):
              rsrcmgr = PDFResourceManager()
              retstr = StringIO()
              codec = 'utf-8'
              laparams = LAParams()
              device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
              fp = file(path, 'rb')
              interpreter = PDFPageInterpreter(rsrcmgr, device)
              password = ""
              maxpages = 0
              caching = True
              pagenos=set()
              for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
                  interpreter.process_page(page)
              text = retstr.getvalue()
              fp.close()
              device.close()
              retstr.close()
              return text
          data = convert_pdf_to_txt(file_path)"""
          filename = t[0] + ".txt"
          call(["abiword", "--to=txt", file_path])
          paperdata = codecs.open(filename,'r', 'utf-8').read()
          if(not paperdata):
              uuid_set = str(uuid.uuid4().fields[-1])[:5]
              with Image(filename=file_path, resolution=200) as img:
                  img.compression_quality = 80
                  img.save(filename="/home/bloomtaxo/fileupload/media/temp%s.jpg" % uuid_set)
                  #pathsave="/home/bloomtaxo/fileupload/media/temp%s.jpg" % uuid_set
                  """image = cv2.imread(pathsave)
                  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                  gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                  filename = "{}.jpg".format(os.getpid())
                  cv2.imwrite(filename, gray)
                  text = pytesseract.image_to_string(PI.open(filename))
                  os.remove(filename)
                  data = text
                  f= open("/home/bloomtaxo/fileupload/media/file.txt","w+")
                  f.write(data)
                  f.close()
                  paperdata=codecs.open("/home/bloomtaxo/fileupload/media/file.txt",'r', 'utf-8').read()
                  return HttpResponse(pathsave)"""
              return HttpResponse('hello')


          questions = []
          #os.remove(file_path)
          #os.remove(filename)
          #f= open("/home/bloomtaxo/fileupload/media/file.txt","w+")
          #f.write(paperdata)
          #f.close()
          #paperdata = codecs.open("/home/bloomtaxo/fileupload/media/file.txt",'r', 'utf-8').read()
          sentences = sent_tokenize(paperdata)
          #os.remove("/home/bloomtaxo/fileupload/media/file.txt")
          for sent in sentences:
              sent= re.sub('\(?[A-Z|a-z|0-9|]\)', '', sent)
              sent = re.sub('\(?(M{0,4}(CM|CD|DC{0,3})(XC|XL|LX{0,3})(IX|IV|VI{0,3})|[IDCXMLV])+\)','',sent)
              sent = re.sub('\(?(m{0,4}(cm|cd|dc{0,3})(xc|xl|lx{0,3})(ix|iv|vi{0,3})|[idcxmlv])+\)','',sent)
              sent = re.sub('\[\"|\']','',sent)
              sent = re.sub('[\r]','',sent)
              sent = re.sub('[0-9][\s]?[M]|[M|m][A|a][R|r][K|k][S|s]','',sent)
              sent = re.sub('\[\[][A-Z|a-z|0-9]+([\(]?([a-z|A-Z|0-9|_|\s|,]+)?[\)]?)?[\]]','',sent)
              se = re.sub('[Q|q]?[\.]?[0-9]+[\.]?','',sent)
              if(se == ""):
                  continue
              if(classifier.classify(get_features((sent.lower())))=="question"):
                  questions.append(sent)
          for ques in questions:
              words = word_tokenize(ques.lower())
              btlevellist = []
              for w in words:
                  for values in bt.values():
                      for keywords in values:
                          if(w==keywords):
                              bta = bt.keys()[bt.values().index(values)]
                              if not bta in btlevellist:
                                  btlevellist.append(bt.keys()[bt.values().index(values)])
              #btperq.append(btlevellist)
              if(len(btlevellist)!=1):
                  btlevellist = []
                  btlevellist.append(btclassifier.classify(get_btfeatures(ques)))


                  #btperq.append(btlevellist)
              #    btperq.append(btclassifier.classify(ques))
                  #if(btlevellist == []):
                  #    btlevellist.append(btclassifier.classify(get_btfeatures(ques)))
                  #else:
                  #    for bts in btlevellist:
                  #        if(nltk.classify.accuracy(btclassifier, [(get_btfeatures(ques.lower()),bts)])==1.0):
                  #            btlevellist = []
                  #            btlevellist.append(bts)

              btperq.append(btlevellist)
          senddata = {'question':questions,'btlevel':btperq,'list':zip(questions, btperq)}
          #senddata = {'question':questions}
          #return Response(senddata, template_name='file.html')
          #return HttpResponse(unicode(pathsave))




          """bt1=['define','describe','draw','find','identify','label','list','locate','match','memorise','name','recall','recite','recognize','relate','reproduce','select','state','tell','write']
          bt2=['compare','convert','demonstarte','describe','discuss','distinguish','explain','find out more information about','generalize','interpret','outline','paraphrase','predict','put into your own words','relate','restate','summarize','translate','visualize']
          bt3=['apply','calculate','change','choose','complete','construct','examine','illustrate','interpret','make','manipulate','modify','produce','put into practice','put together','solve','show','translate','use']
          bt4 = ['advertise','analyse','categoriase','compare','contrast','deduce','differenciate','distinguish','examine','explain','identify','investigate','seperate','subdivide','take apart']
          bt5 = ['argue','assess','choose','compose','construct','create','criticise','critique','debate','decide','defend','design','determine','device','discuss','estimate','evaluate','formulate','imagine','invent','judge','justify','plan','predict','prioritise','propose','rate','recommend','select','value']
          bt6 = ['add to','argue','assess','choose','combine','compose','construct','create','debate','decide','design','determine','devise','discuss','forcast','formulate','hypothesise','imagine','invent','judge','justify','originate','plan','predict','priortise','propose','rate','recommend','select','verify']
          bt = {'bt1': bt1, 'bt2': bt2,'bt3':bt3,'bt4':bt4,'bt5':bt5,'bt6':bt6}
          btperq = []
          data = data.lower()
          sentences = sent_tokenize(data)
          questions = []
          for sen in sentences:
              if(classifier.classify(get_features(sen))=="question"):
                  questions.append(sen)

          for q in questions:
              word_tokens = word_tokenize(q)
              filtered_sentence = [w for w in word_tokens if not w in stop_words]
              btlevellist=[]
              for word in range(len(filtered_sentence)):
                  for values in bt.values():
                      for keywords in values:
                          if(t[word]==keywords):
                              if bt.keys()[bt.values().index(values)] not in btlevellist:
                                  btlevellist.append(bt.keys()[bt.values().index(values)])
              if(len(btlevellist)>1):
                  acc = 0
                  for b in btlevelist:
                      if(nltk.classify.accuracy(classifier1, [(get_features(q),b)])>acc):
                          acc = nltk.classify.accuracy(classifier1, [(get_features(q),b)])
                          btlevellist = b

              btperq.append(btlevellist)
          #senddata = {'question':questions,'btlevel':btperq,'list':zip(questions, btperq)}
          senddata = {'question':data,'list':zip()}
          return Response(senddata, template_name='file.html')"""
      #comment

      if(t[1]=="jpg" or t[1]=="png" or t[1]=="jpeg" or t[1]=="gif"):

          image = cv2.imread(file_path)
          gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
          filename = "{}.png".format(os.getpid())
          cv2.imwrite(filename, gray)
          text = pytesseract.image_to_string(PI.open(filename))
          os.remove(filename)
          data = text
          #data = data.lower()
          """read1 = data.split("Q")
          btperq = []
          for i in range(0,len(read1)):
              btlevellist=[]
              read1[i]=read1[i].translate(digits)
              read1[i] = re.sub('[.,!?]', '', read1[i])
              t = read1[i].split(" ")
              for word in range(len(t)):
                  for values in bt.values():
                      for keywords in values:
                          if(t[word]==keywords):
                               if bt.keys()[bt.values().index(values)] not in btlevellist:
                                   btlevellist.append(bt.keys()[bt.values().index(values)])
              btperq.append(btlevellist)
          senddata = {'question':data,'btlevel':btperq,'list':zip(read1, btperq)}
          return Response(senddata, template_name='file.html')"""
          questions = []
          os.remove(file_path)
          f= codecs.open("/home/bloomtaxo/fileupload/media/file.txt","w+",'utf-8')
          f.write(data)
          f.close()
          paperdata = codecs.open("/home/bloomtaxo/fileupload/media/file.txt",'r', 'utf-8').read()
          sentences = sent_tokenize(paperdata)
          #os.remove("/home/bloomtaxo/fileupload/media/file.txt")
          for sent in sentences:
              sent = re.sub('[\r|\n]','',sent)
              sent = re.sub('[0-9][\s]?[M]|[M|m][A|a][R|r][K|k][S|s]','',sent)
              sent= re.sub('\(?[A-Z|a-z|0-9|]\)', '', sent)
              sent = re.sub('\(?(M{0,4}(CM|CD|DC{0,3})(XC|XL|LX{0,3})(IX|IV|VI{0,3})|[IDCXMLV])+\)','',sent)
              sent = re.sub('\(?(m{0,4}(cm|cd|dc{0,3})(xc|xl|lx{0,3})(ix|iv|vi{0,3})|[idcxmlv])+\)','',sent)
              sent = re.sub('\[\"|\']','',sent)
              sent = re.sub('[\r]','',sent)
              sent = re.sub('[0-9][\s]?[M]|[M|m][A|a][R|r][K|k][S|s]','',sent)
              sent = re.sub('[\[][A-Z|a-z|0-9]+([\(]?([a-z|A-Z|0-9|_|\s|,]+)?[\)]?)?[\]]','',sent)
              sent = re.sub('[Q|q]?[\.]?[0-9]+[\.]?','',sent)
              sent = re.sub('[B|b][t|T][0-9]','',sent)
              sent = re.sub('[B|b][t|T][0-9]?','',sent)
              sent = re.sub('[\n]','',sent)
              sent = re.sub('Question','',sent)
              sent = re.sub('question','',sent)
              if(sent == ""):
                  pass
              else:
                  if(classifier.classify(get_features((sent.lower())))=="question"):
                      questions.append(sent)

              #if(sent == ""):
              #    continue
              #if(classifier.classify(get_features((se.lower())))=="question"):
              #    questions.append(sent)
              for ques in questions:
                  words = word_tokenize(ques.lower())
                  btlevellist = []
                  for w in words:
                      for values in bt.values():
                          for keywords in values:
                              if(w==keywords):
                                  bta = bt.keys()[bt.values().index(values)]
                                  if not bta in btlevellist:
                                      btlevellist.append(bt.keys()[bt.values().index(values)])
                  if(len(btlevellist)!=1):
                      btlevellist = []
                      btlevellist.append(btclassifier.classify(get_btfeatures(ques)))
                  #if(btlevellist == []):
                  #    btlevellist.append(btclassifier.classify(get_btfeatures(ques)))
                  #else:
                  #    for bts in btlevellist:
                  #        if(nltk.classify.accuracy(btclassifier, [(get_btfeatures(ques.lower()),bts)])==1.0):
                  #            btlevellist = []
                  #            btlevellist.append(bts)
                  btperq.append(btlevellist)
          senddata = {'question':questions,'btlevel':btperq,'list':zip(questions, btperq)}
          #senddata = {'question':questions}
          return Response(senddata, template_name='file.html')
          #return HttpResponse(sentences)
      elif(t[1]=='doc' or t[1]=='docx'):
          filename = t[0] + ".txt"
          call(["abiword", "--to=txt", file_path])
          paperdata = codecs.open(filename,'r', 'utf-8').read()
          if(not paperdata):
              text = docx2txt.process(file_path, "/home/bloomtaxo/fileupload/media")
              paperdata = text
              fpath = t[0] + ".txt"
              os.remove(fpath)
              os.remove(file_path)
              """for filename in os.listdir("/home/bloomtaxo/fileupload/media"):
                   img = cv2.imread(os.path.join("/home/bloomtaxo/fileupload/media",filename))
                   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                   gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                   filename = "{}.png".format(os.getpid())
                   cv2.imwrite(filename, gray)
                   text = pytesseract.image_to_string(PI.open(filename))
                   os.remove(filename)
                   paperdata += text"""
              imgs = [i for i in os.listdir("/home/bloomtaxo/fileupload/media") if not os.path.isdir(i)]
              return HttpResponse(imgs)
          sentences = sent_tokenize(paperdata)
          questions =[]
          #os.remove("/home/bloomtaxo/fileupload/media/file.txt")
          for sent in sentences:
              sent= re.sub('\(?[A-Z|a-z|0-9|]\)', '', sent)
              sent = re.sub('\(?(M{0,4}(CM|CD|DC{0,3})(XC|XL|LX{0,3})(IX|IV|VI{0,3})|[IDCXMLV])+\)','',sent)
              sent = re.sub('\(?(m{0,4}(cm|cd|dc{0,3})(xc|xl|lx{0,3})(ix|iv|vi{0,3})|[idcxmlv])+\)','',sent)
              sent = re.sub('\[\"|\']','',sent)
              sent = re.sub('[\r]','',sent)
              sent = re.sub('[0-9][\s]?[M]|[M|m][A|a][R|r][K|k][S|s]','',sent)
              sent = re.sub('\[\[][A-Z|a-z|0-9]+([\(]?([a-z|A-Z|0-9|_|\s|,]+)?[\)]?)?[\]]','',sent)
              se = re.sub('[Q|q]?[\.]?[0-9]+[\.]?','',sent)
              if(se == ""):
                  continue
              if(classifier.classify(get_features((sent.lower())))=="question"):
                  questions.append(sent)
          for ques in questions:
              words = word_tokenize(ques.lower())
              btlevellist = []
              for w in words:
                  for values in bt.values():
                      for keywords in values:
                          if(w==keywords):
                              bta = bt.keys()[bt.values().index(values)]
                              if not bta in btlevellist:
                                  btlevellist.append(bt.keys()[bt.values().index(values)])
              #btperq.append(btlevellist)
              if(len(btlevellist)!=1):
                  btlevellist = []
                  btlevellist.append(btclassifier.classify(get_btfeatures(ques)))


                  #btperq.append(btlevellist)
              #    btperq.append(btclassifier.classify(ques))
                  #if(btlevellist == []):
                  #    btlevellist.append(btclassifier.classify(get_btfeatures(ques)))
                  #else:
                  #    for bts in btlevellist:
                  #        if(nltk.classify.accuracy(btclassifier, [(get_btfeatures(ques.lower()),bts)])==1.0):
                  #            btlevellist = []
                  #            btlevellist.append(bts)

              btperq.append(btlevellist)
          senddata = {'question':questions,'btlevel':btperq,'list':zip(questions, btperq)}
          #senddata = {'question':questions}
          os.remove(file_path)
          os.remove(filename)
          return Response(senddata, template_name='file.html')

          #return HttpResponse(paperdata)



      else:
          data = open(file_path).read()
          questions = []
          os.remove(file_path)
          f= open("/home/bloomtaxo/fileupload/media/file.txt","w+")
          f.write(data)
          f.close()
          paperdata = codecs.open("/home/bloomtaxo/fileupload/media/file.txt",'r', 'utf-8').read()
          sentences = sent_tokenize(paperdata)
          #os.remove("/home/bloomtaxo/fileupload/media/file.txt")
          for sent in sentences:
              sent= re.sub('\(?[A-Z|a-z|0-9|]\)', '', sent)
              sent = re.sub('\(?(M{0,4}(CM|CD|DC{0,3})(XC|XL|LX{0,3})(IX|IV|VI{0,3})|[IDCXMLV])+\)','',sent)
              sent = re.sub('\(?(m{0,4}(cm|cd|dc{0,3})(xc|xl|lx{0,3})(ix|iv|vi{0,3})|[idcxmlv])+\)','',sent)
              sent = re.sub('\[\"|\']','',sent)
              sent = re.sub('[\r]','',sent)
              sent = re.sub('[0-9][\s]?[M]|[M|m][A|a][R|r][K|k][S|s]','',sent)
              sent = re.sub('\[\[][A-Z|a-z|0-9]+([\(]?([a-z|A-Z|0-9|_|\s|,]+)?[\)]?)?[\]]','',sent)
              se = re.sub('[Q|q]?[\.]?[0-9]+[\.]?','',sent)
              if(sent == ""):
                  continue
              if(classifier.classify(get_features((se.lower())))=="question"):
                 questions.append(sent)
          for ques in questions:
              words = word_tokenize(ques.lower())
              btlevellist = []
              for w in words:
                  for values in bt.values():
                      for keywords in values:
                          if(w==keywords):
                              bta = bt.keys()[bt.values().index(values)]
                              if not bta in btlevellist:
                                  btlevellist.append(bt.keys()[bt.values().index(values)])
              if(len(btlevellist)!=1):
                  btlevellist = []
                  btlevellist.append(btclassifier.classify(get_btfeatures(ques)))
                  #if(btlevellist == []):
                  #    btlevellist.append(btclassifier.classify(get_btfeatures(ques)))
                  #else:
                  #    for bts in btlevellist:
                  #        if(nltk.classify.accuracy(btclassifier, [(get_btfeatures(ques.lower()),bts)])==1.0):
                  #            btlevellist = []
                  #            btlevellist.append(bts)

              btperq.append(btlevellist)
          senddata = {'question':questions,'btlevel':btperq,'list':zip(questions, btperq)}
          return Response(senddata, template_name='file.html')
      """bt1=['define','describe','draw','find','identify','label','list','locate','match','memorise','name','recall','recite','recognize','relate','reproduce','select','state','tell','write']
      bt2=['compare','convert','demonstarte','describe','discuss','distinguish','explain','find out more information about','generalize','interpret','outline','paraphrase','predict','put into your own words','relate','restate','summarize','translate','visualize']
      bt3=['apply','calculate','change','choose','complete','construct','examine','illustrate','interpret','make','manipulate','modify','produce','put into practice','put together','solve','show','translate','use']
      bt4 = ['advertise','analyse','categoriase','compare','contrast','deduce','differenciate','distinguish','examine','explain','identify','investigate','seperate','subdivide','take apart']
      bt5 = ['argue','assess','choose','compose','construct','create','criticise','critique','debate','decide','defend','design','determine','device','discuss','estimate','evaluate','formulate','imagine','invent','judge','justify','plan','predict','prioritise','propose','rate','recommend','select','value']
      bt6 = ['add to','argue','assess','choose','combine','compose','construct','create','debate','decide','design','determine','devise','discuss','forcast','formulate','hypothesise','imagine','invent','judge','justify','originate','plan','predict','priortise','propose','rate','recommend','select','verify']
      bt = {'bt1': bt1, 'bt2': bt2,'bt3':bt3,'bt4':bt4,'bt5':bt5,'bt6':bt6}

      #data = data.decode('utf-8','ignore')
      #os.remove(file_path)
      #sentences = sent_tokenize(data.decode('utf-8'))
      #questions = []
      #btperq = []
      #btlevellist = []
      #or sent in sentences:
      #    if(classifier.classify(get_features(sent))=="question"):
      #        questions.append(sent)
      #return HttpResponse(data)
      for ques in questions:
          word_tokens = word_tokenize(unicode(ques))
          for word in word_tokens:
                  for values in bt.values():
                      for keywords in values:
                          if(word==keywords):
                               if bt.keys()[bt.values().index(values)] not in btlevellist:
                                   btlevellist.append(bt.keys()[bt.values().index(values)])
          btperq.append(btlevellist)
      senddata = {'question':questions,'btlevel':btperq,'list':zip(questions, btperq)}

      return Response(senddata, template_name='file.html')"""








    else:
      return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class File1(APIView):
    renderer_classes = [TemplateHTMLRenderer]
    template_name = 'file.html'

    def get(self, request):

        return Response("hello")

