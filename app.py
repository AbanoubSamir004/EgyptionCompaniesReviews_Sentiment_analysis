from importlib.resources import path
import streamlit as st 
import pandas as pd
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode
from transformers import pipeline
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import PIL
from PIL import Image
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import qalsadi.lemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import re
import emoji
import string
import numpy as np
import joblib as jb
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression

vectorizer=TfidfVectorizer(max_features=1000,ngram_range=(1, 2))

def arabic_trained_model():
    preprocessing_data=pd.read_excel("final_preprocessing_dataset.xlsx")
    preprocessing_data.dropna(subset=['final_text_lemmatizer'], how='any', inplace=True)

    unigramdataGet= vectorizer.fit_transform(preprocessing_data['final_text_lemmatizer'].astype('str'))
    unigramdataGet = unigramdataGet.toarray()
    vocab = vectorizer.get_feature_names()
    unigramdata_features=pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
    unigramdata_features[unigramdata_features>0] = 1

    Y=preprocessing_data.rating
    arabic_train_model = LogisticRegression().fit(unigramdata_features, Y)
    return arabic_train_model

    
train_model=arabic_trained_model()



def getX(df):
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    stop_words = list(set(stopwords.words('arabic')))
    print(stop_words)
    stop_words.remove('لا')
    stop_words.remove('لكن')
    stop_words.remove('ولكن')
    stop_words.remove('واو')
    stop_words.remove('أطعم')
    stop_words.remove('أف')
    def remove_diacritics(text):
        arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
        text = re.sub(arabic_diacritics, '', str(text))
        return text

    def remove_emoji(text):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',text)

    def clean_text(text):
        text = "".join([word for word in text if word not in string.punctuation])
        text = remove_emoji(text)
        text = remove_diacritics(text)
        tokens = word_tokenize(text)
        text = ' '.join([word for word in tokens if word not in stop_words])
        return text

    df.review_description= df.review_description.astype(str)
    df['cleaned_text'] = df.review_description.apply(clean_text)

    df.dropna(inplace=True)
    df.drop_duplicates(subset=['cleaned_text'],inplace=True)


    lemmer = qalsadi.lemmatizer.Lemmatizer()
    df['final_text'] = df.cleaned_text.apply(lambda x:lemmer.lemmatize_text(x))

    def convert_list_to_str(data):
        data = str(data)
        data = data.replace("'",'')
        data = data.replace(',','')                                                                                     
        data = data.replace('[','')
        data = data.replace(']','')

        return data

    df['final_text'] = df.final_text.apply(convert_list_to_str)
    df.to_excel('final_lemmatization.xlsx')

    text_features=vectorizer.transform(df["final_text"])
    my_array=text_features.toarray()
    X=pd.DataFrame(my_array,columns=vectorizer.get_feature_names())
    return X

def setup_model():  
    """
    Setup Model
    """
    # this will download 2 GB
    nlp = pipeline("sentiment-analysis", model='XLM-R-L-ARABIC-SENT')
    return nlp

def get_tweets(query,limit):
    """
    Get Tweets
    """
    #query = '"IBM" min_replies:10 min_faves:500 min_retweets:10 lang:ar'
    tweets = []
    pbar = st.progress(0)
    latest_iteration = st.empty()
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        latest_iteration.text(f'{int(len(tweets)/limit*100)}% Done')
        pbar.progress(int(len(tweets)/limit*100))
        if len(tweets)==limit:
            break
        else:
            tweets.append(tweet.content)
    return tweets


# Fxn
def convert_to_df(sentiment):
    """
    Convert to df
    """
    label2 = sentiment[1] if 1 in sentiment.index else 0
    label1 = sentiment[-1] if -1 in sentiment.index else 0
    label0 = sentiment[0] if 0 in sentiment.index else 0
    sentiment_dict = {'Positive':label2,'Neutral':label0,'Negative':label1}
    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['Sentiment','Count'])
    return sentiment_df

emojis = {
    "🙂":"يبتسم",
    "😂":"يضحك",
    "💔":"قلب حزين",
    "🙂":"يبتسم",
    "❤️":"حب",
    "❤":"حب",
    "😍":"حب",
    "😭":"يبكي",
    "😢":"حزن",
    "😔":"حزن",
    "♥":"حب",
    "💜":"حب",
    "😅":"يضحك",
    "🙁":"حزين",
    "💕":"حب",
    "💙":"حب",
    "😞":"حزين",
    "😊":"سعادة",
    "👏":"يصفق",
    "👌":"احسنت",
    "😴":"ينام",
    "😀":"يضحك",
    "😌":"حزين",
    "🌹":"وردة",
    "🙈":"حب",
    "😄":"يضحك",
    "😐":"محايد",
    "✌":"منتصر",
    "✨":"نجمه",
    "🤔":"تفكير",
    "😏":"يستهزء",
    "😒":"يستهزء",
    "🙄":"ملل",
    "😕":"عصبية",
    "😃":"يضحك",
    "🌸":"وردة",
    "😓":"حزن",
    "💞":"حب",
    "💗":"حب",
    "😑":"منزعج",
    "💭":"تفكير",
    "😎":"ثقة",
    "💛":"حب",
    "😩":"حزين",
    "💪":"عضلات",
    "👍":"موافق",
    "🙏🏻":"رجاء طلب",
    "😳":"مصدوم",
    "👏🏼":"تصفيق",
    "🎶":"موسيقي",
    "🌚":"صمت",
    "💚":"حب",
    "🙏":"رجاء طلب",
    "💘":"حب",
    "🍃":"سلام",
    "☺":"يضحك",
    "🐸":"ضفدع",
    "😶":"مصدوم",
    "✌️":"مرح",
    "✋🏻":"توقف",
    "😉":"غمزة",
    "🌷":"حب",
    "🙃":"مبتسم",
    "😫":"حزين",
    "😨":"مصدوم",
    "🎼 ":"موسيقي",
    "🍁":"مرح",
    "🍂":"مرح",
    "💟":"حب",
    "😪":"حزن",
    "😆":"يضحك",
    "😣":"استياء",
    "☺️":"حب",
    "😱":"كارثة",
    "😁":"يضحك",
    "😖":"استياء",
    "🏃🏼":"يجري",
    "😡":"غضب",
    "🚶":"يسير",
    "🤕":"مرض",
    "‼️":"تعجب",
    "🕊":"طائر",
    "👌🏻":"احسنت",
    "❣":"حب",
    "🙊":"مصدوم",
    "💃":"سعادة مرح",
    "💃🏼":"سعادة مرح",
    "😜":"مرح",
    "👊":"ضربة",
    "😟":"استياء",
    "💖":"حب",
    "😥":"حزن",
    "🎻":"موسيقي",
    "✒":"يكتب",
    "🚶🏻":"يسير",
    "💎":"الماظ",
    "😷":"وباء مرض",
    "☝":"واحد",
    "🚬":"تدخين",
    "💐" : "ورد",
    "🌞" : "شمس",
    "👆" : "الاول",
    "⚠️" :"تحذير",
    "🤗" : "احتواء",
    "✖️": "غلط",
    "📍"  : "مكان",
    "👸" : "ملكه",
    "👑" : "تاج",
    "✔️" : "صح",
    "💌": "قلب",
    "😲" : "مندهش",
    "💦": "ماء",
    "🚫" : "خطا",
    "👏🏻" : "برافو",
    "🏊" :"يسبح",
    "👍🏻": "تمام",
    "⭕️" :"دائره كبيره",
    "🎷" : "ساكسفون",
    "👋": "تلويح باليد",
    "✌🏼": "علامه النصر",
    "🌝":"مبتسم",
    "➿"  : "عقده مزدوجه",
    "💪🏼" : "قوي",
    "📩":  "تواصل معي",
    "☕️": "قهوه",
    "😧" : "قلق و صدمة",
    "🗨": "رسالة",   
    "❗️" :"تعجب",
    "🙆🏻": "اشاره موافقه",
    "👯" :"اخوات",
    "©" :  "رمز",
    "👵🏽" :"سيده عجوزه",
    "🐣": "كتكوت",  
    "🙌": "تشجيع",
    "🙇": "شخص ينحني",
    "👐🏽":"ايدي مفتوحه",    
    "👌🏽": "بالظبط",
    "⁉️" : "استنكار",
    "⚽️": "كوره",
    "🕶" :"حب",
    "🎈" :"بالون",    
    "🎀":    "ورده",
    "💵":  "فلوس",   
    "😋":  "جائع",
    "😛":  "يغيظ",
    "😠":  "غاضب",
    "✍🏻":  "يكتب",
    "🌾":  "ارز",
    "👣":  "اثر قدمين",
    "❌":"رفض",
    "🍟":"طعام",
    "👬":"صداقة",
    "🐰":"ارنب",
    "☂":"مطر",
    "⚜":"مملكة فرنسا",
    "🐑":"خروف",
    "🗣":"صوت مرتفع",
    "👌🏼":"احسنت",
    "☘":"مرح",
    "😮":"صدمة",
    "😦":"قلق",
    "⭕":"الحق",
    "✏️":"قلم",
    "ℹ":"معلومات",
    "🙍🏻":"رفض",
    "⚪️":"نضارة نقاء",
    "🐤":"حزن",
    "💫":"مرح",
    "💝":"حب",
    "🍔":"طعام",
    "❤︎":"حب",
    "✈️":"سفر",
    "🏃🏻‍♀️":"يسير",
    "🍳":"ذكر",
    "🎤":"مايك غناء",
    "🎾":"كره",
    "🐔":"دجاجة",
    "🙋":"سؤال",
    "📮":"بحر",
    "💉":"دواء",
    "🙏🏼":"رجاء طلب",
    "💂🏿 ":"حارس",
    "🎬":"سينما",
    "♦️":"مرح",
    "💡":"قكرة",
    "‼":"تعجب",
    "👼":"طفل",
    "🔑":"مفتاح",
    "♥️":"حب",
    "🕋":"كعبة",
    "🐓":"دجاجة",
    "💩":"معترض",
    "👽":"فضائي",
    "☔️":"مطر",
    "🍷":"عصير",
    "🌟":"نجمة",
    "☁️":"سحب",
    "👃":"معترض",
    "🌺":"مرح",
    "🔪":"سكينة",
    "♨":"سخونية",
    "👊🏼":"ضرب",
    "✏":"قلم",
    "🚶🏾‍♀️":"يسير",
    "👊":"ضربة",
    "◾️":"وقف",
    "😚":"حب",
    "🔸":"مرح",
    "👎🏻":"لا يعجبني",
    "👊🏽":"ضربة",
    "😙":"حب",
    "🎥":"تصوير",
    "👉":"جذب انتباه",
    "👏🏽":"يصفق",
    "💪🏻":"عضلات",
    "🏴":"اسود",
    "🔥":"حريق",  
    "😬":"عدم الراحة",   
    "👊🏿":"يضرب",    
    "🌿":"ورقه شجره",     
    "✋🏼":"كف ايد",    
    "👐":"ايدي مفتوحه",      
    "☠️":"وجه مرعب",     
    "🎉":"يهنئ",      
    "🔕" :"صامت",
    "😿":"وجه حزين",      
    "☹️":"وجه يائس",     
    "😘" :"حب",     
    "😰" :"خوف و حزن",
    "🌼":"ورده",      
    "💋":  "بوسه",
    "👇":"لاسفل",     
    "❣️":"حب",     
    "🎧":"سماعات",
    "📝":"يكتب",      
    "😇":"دايخ",      
    "😈":"رعب",      
    "🏃":"يجري",      
    "✌🏻":"علامه النصر",    
    "🔫":"يضرب",      
    "❗️":"تعجب",
    "👎":"غير موافق",      
    "🔐":"قفل",      
    "👈":"لليمين",
    "™":"رمز",    
    "🚶🏽":"يتمشي",    
    "😯":"متفاجأ",  
    "✊":"يد مغلقه",    
    "😻":"اعجاب",    
    "🙉" :"قرد",    
    "👧":"طفله صغيره",     
    "🔴":"دائره حمراء",      
    "💪🏽":"قوه",     
    "💤":"ينام",     
    "👀":"ينظر",     
    "✍🏻":"يكتب",  
    "❄️":"تلج",
    "💀":"رعب",   
    "😤":"وجه عابس",      
    "🖋":"قلم",      
    "🎩":"كاب",      
    "☕️":"قهوه",     
    "😹":"ضحك",     
    "💓":"حب",      
    "☄️ ":"نار",     
    "👻":"رعب",
    "❎":"خطء",
    "🤮":"حزن",
    '🏻':"احمر"
}

emoticons_to_emoji = {
    ":)" : "🙂",
    ":(" : "🙁",
    "xD" : "😆",
    ":=(": "😭",
    ":'(": "😢",
    ":'‑(": "😢",
    "XD" : "😂",
    ":D" : "🙂",
    "♬" : "موسيقي",
    "♡" : "❤",
    "☻"  : "🙂",
}

def data_preprocessing(data):
 
    X = getX(data)
    sentiment = train_model.predict(X)
    return sentiment

def get_sentiments(tweets,opt):
    """
    Get sentiments
    """
    if opt =='Arabic':
        sentiments = []
        pbar = st.progress(0)
        latest_iteration = st.empty()
        tweets=pd.DataFrame(tweets,columns=['review_description'])
        sentiments=data_preprocessing(tweets)
        x=pd.DataFrame(sentiments)
        x.to_excel('counts.xlsx')
    else:
        nlp = setup_model()
        sentiments = []
        pbar = st.progress(0)
        latest_iteration = st.empty()
        for i,tweet in enumerate(tweets):
            latest_iteration.text(f'{int((i+1)/len(tweets)*100)}% Done')
            pbar.progress(int((i+1)/len(tweets)*100))
            sentiments.append(nlp(tweet)[0]['label'])
    return sentiments

def main():
    """
    Main
    """
    st.set_page_config(layout="wide",initial_sidebar_state="collapsed")
    col1,col2,col3 = st.columns((1,2,1))
    image = Image.open('banquemisr.png')
    #smileys = Image.open('smileys.png')
    with col3:
        st.image(image)
    with col1:
        st.title("Company Based Sentiment")
        st.subheader("Made by Fahd Seddik")
    with col2:
        st.title("")
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        with col1:
            st.subheader("Home")
            
            temp = st.slider("Choose sample size",min_value=50,max_value=10000)
            opt = st.selectbox('Language',pd.Series(['Arabic','English']))
            likes = st.slider("Minimum number of likes on tweet",min_value=0,max_value=5000)
            retweets = st.slider("Minimum number of retweets on tweet",min_value=0,max_value=500)
            with st.form(key='nlpForm'):
                raw_text = st.text_area("Enter company name here")
                submit_button = st.form_submit_button(label='Analyze')       
        # layout
        if submit_button:
            with col2:
                st.success(f'Selected Language: {opt}',)
                if opt=='Arabic':
                    query = raw_text + ' lang:ar'
                else:
                    query = raw_text + ' lang:en'
                query += f' min_faves:{likes} min_retweets:{retweets}'
                st.write('Retreiving Tweets...')
                tweets = get_tweets(query,temp)
                if(len(tweets)==temp):
                    st.success(f'Found {len(tweets)}/{temp}')
                else:
                    st.error(f'Only Found {len(tweets)}/{temp}. Try changing min_likes, min_retweets')
                st.write('Loading Model...')
                #nlp = setup_model()
                st.success('Loaded Model')
                st.write('Analyzing Sentiments...')
                sentiments = get_sentiments(tweets,opt)
                st.success('DONE')
                st.subheader("Results of "+raw_text)
                counts = pd.Series(sentiments).value_counts()
                result_df = convert_to_df(counts)
                # Dataframe
                st.dataframe(result_df)

                fig1, ax1 = plt.subplots(figsize=(5,5))
                ax1.pie(result_df['Count'],labels=result_df['Sentiment'].values, autopct='%1.1f%%',
                        shadow=True, startangle=90,colors=['green','yellow','red'])
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig1)
                tweets_s = pd.Series(tweets,name='Tweet')
                sentiments_s = pd.Series(sentiments,name='Sentiment (pred)').replace(
                    {'LABEL_2':'Positive','LABEL_1':'Negative','LABEL_0':'Neutral'})
                all_df = pd.merge(left=tweets_s,right=sentiments_s,left_index=True,right_index=True)
                all_df.to_excel('tweets.xlsx')
            st.subheader("Tweets")
            gb = GridOptionsBuilder.from_dataframe(all_df)
            gb.configure_side_bar()
            grid_options = gb.build()
            AgGrid(
                all_df,
                gridOptions=grid_options,
                enable_enterprise_modules=True,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                fit_columns_on_grid_load=False,
            )
            with col2:
                st.subheader("Word Cloud")
                #Word Cloud
                if opt=='English':
                    words = " ".join(word for tweet in tweets for word in tweet.split())
                    st_en = set(stopwords.words('english'))
                    #st_ar = set(stopwords.words('arabic'))
                    st_en = st_en.union(STOPWORDS).union(set(['https','http','DM','dm','via','co']))
                    wordcloud = WordCloud(stopwords=st_en, background_color="white", width=800, height=400)
                    wordcloud.generate(words)
                    fig2, ax2 = plt.subplots(figsize=(5,5))
                    ax2.axis("off")
                    fig2.tight_layout(pad=0)
                    ax2.imshow(wordcloud, interpolation='bilinear')
                    st.pyplot(fig2)
                else:
                    st.error(f'WordCloud not available for Language = {opt}')

    else:
        st.subheader("About")
        st.write("This was made in order to have an idea about people's opinion on a certain company. The program scrapes twitter for tweets\
             that are about a certain company. The tweets are then fed into a model for sentiment analysis which is then used \
            to display useful information about each company and public opinion.")


if __name__ == '__main__':
	main()