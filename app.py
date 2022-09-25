"""
Streamlit app
"""

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
import re
import emoji
from sklearn.svm import LinearSVC
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


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

def arabic_trained_model():
    preprocessing_data=pd.read_excel("final_preprocessing_dataset.xlsx")
    preprocessing_data.dropna(subset=['final_text_lemmatizer'], how='any', inplace=True)

    word_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
    unigramdataGet= word_vectorizer.fit_transform(preprocessing_data['final_text_lemmatizer'].astype('str'))
    unigramdataGet = unigramdataGet.toarray()
    vocab = word_vectorizer.get_feature_names()
    unigramdata_features=pd.DataFrame(np.round(unigramdataGet, 1), columns=vocab)
    unigramdata_features[unigramdata_features>0] = 1

    Y=preprocessing_data.rating
    arabic_train_model = LogisticRegression().fit(unigramdata_features, Y)
    return arabic_train_model


train_model=joblib.load("LogisticRegression")



def data_preprocessing(tweet):
 
    stopWords=list(set(stopwords.words("arabic")))## To remove duplictes and return to list again 
    #Some words needed to work with to will remove 
    for word in ['واو','لا','لكن','ولكن','أطعم', 'أف','ليس','ولا','ما']:
        stopWords.remove(word)
  #Remove Punctuation
    tweet['clean_text']=tweet['reviews'].astype(str)
    tweet['clean_text']=tweet['clean_text'].apply(lambda x:re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', x))
    tweet['clean_text']=tweet['clean_text'].apply(lambda x:x.replace('؛',"", ))
    #remove stopwords
    tweet['clean_text']=tweet['clean_text'].apply(lambda x:" ".join([word for word in x.split() if word not in stopWords]))
    #Handle Emojies
    tweet['clean_text']=tweet['clean_text'].apply(lambda x:emojiTextTransform(x))
    #Remove Numbers
    tweet['clean_text']=tweet['clean_text'].apply(lambda x:''.join([word for word in x if not word.isdigit()]))

    tweet.dropna(inplace=True)
    tweet.drop_duplicates(subset=['clean_text'],inplace=True)

    lemmer = qalsadi.lemmatizer.Lemmatizer()
    tweet['data_lemmatization'] = tweet.clean_text.apply(lambda x:lemmer.lemmatize_text(x))
    tweet['data_lemmatization']=tweet.data_lemmatization.apply(lambda x:" ".join(x))

    vectorizer=TfidfVectorizer(max_features=1000,ngram_range=(1, 2))
    FeatureText=vectorizer.fit_transform(tweet.data_lemmatization)
    X=pd.DataFrame(FeatureText.toarray(),columns=vectorizer.get_feature_names_out())
    sentiment=train_model.predict(X)
    return sentiment

def checkemojie(text):
    emojistext=[]
    for char in text:
        if any(emoji.distinct_emoji_list(char)) and char in emojis.keys():
            emojistext.append(emojis[emoji.distinct_emoji_list(char)[0]])
    return " ".join(emojistext)
    
def emojiTextTransform(text):
    cleantext=re.sub(r'[^\w\s]','',text)
    return cleantext+" "+checkemojie(text)

#calling trained model

def get_sentiments(tweets,opt):
    """
    Get sentiments
    """
    if opt =='Arabic':
        sentiments = []
        pbar = st.progress(0)
        latest_iteration = st.empty()
        #tweets=pd.DataFrame(tweets,columns=['reviews'])
        tweets=pd.read_excel("E:\Company-Sentiment-Analysis-master\capiter.xlsx")
        tweets.rename(columns = {'review_description':'reviews'}, inplace = True)
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
            #st.write("Number of Samples: ",temp)
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