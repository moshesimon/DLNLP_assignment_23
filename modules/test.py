a = {'א':"'",'ב':'b','ג':'g','ד':'d','ה':'h','ו':'w','ז':'z','ח':'.h','ט':'.t','י':'y','כ':'k','ך':'K','ל':'l','מ':'m','נ':'n','ס':'s','ע':'`','פ':'p','צ':'.s','ק':'q','ר':'r','ש':'/s','ת':'t','ם':'M','ן':'N','ץ':'.S','ף':'P','ך':'K'}

def transliterate(word):
    for k,v in a.items():
        word = word.replace(k,v)
    return word

print(transliterate('סוף דבר הכל נשמע, את האלהים ירא ואת מצוותיו שמור, כי זה כל האדם'))