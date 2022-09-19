# pip install playsound==1.2.2
# pip install gTTS


from gtts import gTTS

file_name = 'sample.mp3'
# 영어 문장
'''
text = 'Can I help you?'
tts_en = gTTS(text=text,lang='en')
tts_en.save(file_name)
'''

# 한글 문장
'''
text = '파이썬 좋아'
tts_en = gTTS(text=text,lang='ko')
tts_en.save(file_name)
'''

# 긴 문장 (파일에서 불러와서 처리)
with open('sample.txt','r',encoding='utf8') as f:
    text = f.read()


# mp3 실행
from playsound import playsound
playsound(file_name) # 파일 경로 조정 
