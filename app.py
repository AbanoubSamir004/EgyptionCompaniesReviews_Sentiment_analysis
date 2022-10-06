"""
this section is mainly for import packages for GUI, Tokenization, Stemming, text proprocessing and modeling 
"""
import webbrowser
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
from PIL import Image
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import qalsadi.lemmatizer
import re
import emoji
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
import nltk
from tensorflow import keras

nltk.download('stopwords')
#please download the pretrained model for english sentiments [note: Only download it once. ]
#nlp = pipeline("sentiment-analysis", model='akhooli/xlm-r-large-arabic-sent')
#nlp.save_pretrained('XLM-R-L-ARABIC-SENT')



#this part is defining each emojie as a word descriping its meaning
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


def preprocess_arabic(df):
    # defining a vectorizer with max_features as 1000 and ngrams as (1, 2) 
    # which will be used for converting text into numerical form so the ML algorithm can handel it
    filename='Custom trained model\custom_vectorizer.pk'
    vectorizer = pickle.load(open(filename,'rb'))

    #this function renoves the diacritics(التشكيل او الزخارف)
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

    #checks the existance of emojis
    def checkemojie(text):
        emojistext=[]
        for char in text:
            if any(emoji.distinct_emoji_list(char)) and char in emojis.keys():
                emojistext.append(emojis[emoji.distinct_emoji_list(char)[0]])
        return " ".join(emojistext)

    #subestitute each emoji with its arabic meaning
    def emojiTextTransform(text):
        cleantext=re.sub(r'[^\w\s]','',text)
        return cleantext+" "+checkemojie(text)

    # in this function we abily the bast functions to preprocess text perfore vecotization
    def clean_text(text):
 
        stopWords=list(set(stopwords.words("arabic")))## To remove duplictes and return to list again 
        #Some words needed to work with to will remove 
        for word in ['واو','لا','لكن','ولكن','أطعم', 'أف','ليس','ولا','ما']:
            stopWords.remove(word)
        #Remove Punctuation
        text['cleaned_text']=text['review_description'].astype(str)
        text['cleaned_text']=text['cleaned_text'].apply(lambda x:re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', x))
        text['cleaned_text']=text['cleaned_text'].apply(lambda x:x.replace('؛',"", ))
        #Handle Emojies
        text['cleaned_text']=text['cleaned_text'].apply(lambda x:emojiTextTransform(x))
        #Remove diacritics
        text['cleaned_text']=df.cleaned_text.apply(remove_diacritics)
        #remove stopwords
        text['cleaned_text']=text['cleaned_text'].apply(lambda x:" ".join([word for word in x.split() if word not in stopWords]))
        #Remove Numbers
        text['cleaned_text']=text['cleaned_text'].apply(lambda x:''.join([word for word in x if not word.isdigit()]))

        #remove nulls and duplicated
        text.dropna(inplace=True)
        text.drop_duplicates(subset=['cleaned_text'],inplace=True)
        return text

     #calling clean_text
    df = clean_text(df)

    #lemmatizing text
    lemmer = qalsadi.lemmatizer.Lemmatizer()
    df['final_text'] = df.cleaned_text.apply(lambda x:lemmer.lemmatize_text(x))

    #converts each list of words to a string 
    def convert_list_to_str(data):
        data = str(data)
        data = data.replace("'",'')
        data = data.replace(',','')                                                                                     
        data = data.replace('[','')
        data = data.replace(']','')
        return data

    #applying text conversion
    df['final_text'] = df.final_text.apply(convert_list_to_str)

    #vectorize all text to numerical vectors and returning it as features
    text_features=vectorizer.transform(df["final_text"])
    my_array=text_features.toarray()
    X=pd.DataFrame(my_array,columns=vectorizer.get_feature_names_out())
    return X

def trained_model(lang):  
    """
        trained_model Model
        Output:
            nlp: XLM RoBERTa HuggingFace multilingual Model
            or
            our arabic trained model
    """
    if lang=="Arabic":
        #used logistic regression model
        filename = 'Custom trained model\logistic_regression_trained_model.sav'
        #load dense model 
        #loaded_model = keras.models.load_model(filename)
        loaded_model = pickle.load(open(filename, 'rb'))
    else:
        # MAKE SURE the model is save in 'XLM-R-L-ARABIC-SENT'
        # model link: https://huggingface.co/akhooli/xlm-r-large-arabic-sent
        loaded_model = pipeline("sentiment-analysis", model='XLM-R-L-ARABIC-SENT')
    return loaded_model


def get_tweets(query,limit):
    """
    Get Tweets
    Input: 
        query:  ex.  query = '"IBM" min_replies:10 min_faves:500 min_retweets:10 lang:ar'
        limit: number of tweets
    Output:
        tweets: scraped tweets from twitter
    """
    #
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


#store fetched sentiments in a data frame
def convert_to_df(sentiment_counts,opt):
    """
    Convert to df
    Input:
        sentiment_counts: list of sentiment counts
        opt: language - Arabic/English
    Output:
        sentiment_df: dataframe of mapped sentiments
    """
    if opt=='Arabic':
        label2 = sentiment_counts[1] if 1 in sentiment_counts.index else 0
        label1 = sentiment_counts[-1] if -1 in sentiment_counts.index else 0
        label0 = sentiment_counts[0] if 0 in sentiment_counts.index else 0
        sentiment_dict = {'Positive':label2,'Neutral':label0,'Negative':label1}
    else:
        label2 = sentiment_counts.LABEL_2 if 'LABEL_2' in sentiment_counts.index else 0
        label1 = sentiment_counts.LABEL_1 if 'LABEL_1' in sentiment_counts.index else 0
        label0 = sentiment_counts.LABEL_0 if 'LABEL_0' in sentiment_counts.index else 0
        sentiment_dict = {'Positive':label2,'Neutral':label0,'Negative':label1}

    sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['Sentiment','Count'])
    return sentiment_df

def getCats(y_hat):
    """
    getCats: used to convert softmax output to one hot [Used only with dense models]
    Input:
        y_hat: softmax output 
        opt: one hot output
    Output:
        cat_list: one hot output
    """
    cat_list = []
    for i in y_hat:
        cat_list.append(np.argmax(i) - 1)
    return np.array(cat_list)

def get_sentiments(tweets,opt,model):
    """
    Get sentiments
    Input:
        tweets: fetched tweets
        opt: language - Arabic/English
        model: loaded model
    Output:
        sentiments: predicted sentiments by model
    """
    sentiments = []
    pbar = st.progress(0)
    latest_iteration = st.empty()
    if opt =='Arabic':
        tweets=pd.DataFrame(tweets,columns=['review_description'])
        X=preprocess_arabic(tweets)
        sentiments = model.predict(X)
        #for dense model only
        #sentiments = getCats(sentiments)
        pbar.progress(100)
        latest_iteration.text('100% Done')
            
    else:
        for i,tweet in enumerate(tweets):
            latest_iteration.text(f'{int((i+1)/len(tweets)*100)}% Done')
            pbar.progress(int((i+1)/len(tweets)*100))
            sentiments.append(model(tweet)[0]['label'])
    return sentiments

# MAIN GUI
def main():
    """
    Main
    """
    st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

    #split the page into 3 sections [col1, col2, col3]
    col1,col2,col3 = st.columns((1,2,1))
    image = Image.open('banquemisr.png')

    #put banque misr image into col3
    with col3:
        st.image(image)
    with col1:
        st.title("Company Based Sentiment")
    with col2:
        st.title("")
    
    #make a menu with three options
    menu = ["Home","Data Visualization","About"]
    choice = st.sidebar.selectbox("Menu",menu,key='menu_bar')

    #selected the Home option
    if choice == "Home":
        #boolean variable used for english word cloud and diplay the tweets
        flag=0
        
        #put this action in col1
        with col1:
            st.subheader("Home")
            
            #Choose the sample size for the tweets.
            temp = st.slider("Choose sample size",min_value=50,max_value=100000)

            #choose the language model:
            #if Arabic, then we use our arabic trained model
            #if english, then we use XLM RoBERTa HuggingFace multilingual Model
            opt = st.selectbox('Language',pd.Series(['Arabic','English']))
            
            #Choose the Minimum number of retweets for the tweets.
            likes = st.slider("Minimum number of likes on tweet",min_value=0,max_value=50000)

            #Choose the Minimum number of likes for the tweets.
            retweets = st.slider("Minimum number of retweets on tweet",min_value=0,max_value=5000)

        #selected the arabic language/ model
        if opt=="Arabic":

            #put this action in col1
            with col1:
                #create a form to take the company name in [english/arabic] or upload the dataset file
                with st.form(key='form',clear_on_submit=True):
                    fileTypes = ["csv", "xlsx"]
                    comp_en = st.text_input("Company name (English)",key='comp_en')
                    comp_ar = st.text_input("Company name (Arabic)",key='comp_ar')
                    file = st.file_uploader("Upload file", type=fileTypes)
                    submit_button = st.form_submit_button(label='Analyze')

            # layout
            #submitted "Analyze button"
            if submit_button:
                check=0
                #put this action in col2
                with col2:
                    #company names written successfully.
                    if len(comp_en)>0 and len(comp_ar)>0:
                        flag=1 
                        check=1
                        st.success(f'Selected Language: Arabic, English')

                        #create Twitter queries.
                        query_en = comp_en + ' lang:ar'
                        query_ar = comp_ar +' lang:ar'
                        query_en += f' min_faves:{likes} min_retweets:{retweets}'
                        query_ar += f' min_faves:{likes} min_retweets:{retweets}'

                        #scraping the arabic reviews for the english company name
                        st.write(f'Retreiving Tweets [{comp_en}]...')
                        #collect data from twitter
                        tweets_en = get_tweets(query_en,temp)

                        #Checking how many tweets are found
                        if(len(tweets_en)==temp):
                            st.success(f'Found {len(tweets_en)}/{temp}')
                        else:
                            st.error(f'Only Found {len(tweets_en)}/{temp}. Try changing min_likes, min_retweets')
                        
                        #scraping the arabic reviews for the arabic company name
                        st.write(f'Retreiving Tweets [{comp_ar}]...')
                        
                        #collect data from twitter
                        tweets_ar = get_tweets(query_ar,temp)

                        #Checking how many tweets are found
                        if(len(tweets_ar)==temp):
                            st.success(f'Found {len(tweets_ar)}/{temp}')
                        else:
                            st.error(f'Only Found {len(tweets_ar)}/{temp}. Try changing min_likes, min_retweets')

                        #Merging tweets
                        tweets_ar=pd.DataFrame(tweets_ar,columns=['review_description'])
                        tweets_en=pd.DataFrame(tweets_en,columns=['review_description'])
                        tweets=pd.concat([tweets_ar,tweets_en])
                        comp_name=comp_en+comp_ar

                    #uploaded dataset file successfully
                    elif file:
                        flag=1 
                        check=1
                        st.success(f"[{file.name}] Uploaded Successfully...")

                        #cCheck the dataset file types [csv/xlsx]
                        if file.type=='csv':
                            tweets = pd.read_csv(file)
                        else:
                            tweets = pd.read_excel(file)

                        comp_name=file.name

                    #If the user uploads the dataset file or writes the company name
                    if check==1:
                        #loading model
                        st.write('Loading Model...')
                        model= trained_model("Arabic")
                        st.success('Loaded Model')
                        #analyzing data
                        st.write('Analyzing Sentiments...')
                        #preprocessing and model prediction
                        sentiments = get_sentiments(tweets,'Arabic',model)

                        st.success('DONE')
                        st.subheader(f"Results of {comp_name}")

                        #count [Positive - negative - neutral] results
                        counts = pd.Series(sentiments).value_counts()

                        result_df = convert_to_df(counts,"Arabic")
                    
                        # display the Dataframe result
                        st.dataframe(result_df)

                        #plot a pie chart for the results   
                        fig1, ax1 = plt.subplots(figsize=(5,5))
                        ax1.pie(result_df['Count'],labels=result_df['Sentiment'].values, autopct='%1.1f%%',
                                shadow=True, startangle=90,colors=['green','yellow','red'])
                        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.    
                        st.pyplot(fig1)

                        #Merging tweets and its sentiments prediction results 
                        sentiments_s = pd.Series(sentiments,name='Sentiment (pred)').replace(
                            {1:'Positive',0:'Neutral',-1:'Negative'})
                        all_df = pd.merge(left=tweets,right=sentiments_s,left_index=True,right_index=True)
                        check=0

        #selected the english language/model              
        else:
            #put this action in col1
            with col1:
                #create a form to take the english company name 
                with st.form(key='nlpForm'):
                    raw_text = st.text_area("Enter company name here")
                    submit_button = st.form_submit_button(label='Analyze')   

            #submitted "Analyze button" and company name written successfully.
            if submit_button and len(raw_text)>0:
                flag=1
                #put this action in col2
                with col2:
                    st.success(f'Selected English Sentiment Model')    
                    #create Twitter query.
                    query = raw_text + ' lang:en'
                    query += f' min_faves:{likes} min_retweets:{retweets}'

                    #scraping the english reviews
                    st.write('Retreiving Tweets...')

                    #collect data from twitter
                    tweets = get_tweets(query,temp)

                    #Checking how many tweets are found
                    if(len(tweets)==temp):
                        st.success(f'Found {len(tweets)}/{temp}')
                    else:
                        st.error(f'Only Found {len(tweets)}/{temp}. Try changing min_likes, min_retweets')

                    #loading model    
                    st.write('Loading Model...')
                    #selected XLM RoBERTa HuggingFace multilingual Model
                    model_en = trained_model("English")
                    st.success('Loaded Model')

                    #preprocessing and model prediction
                    st.write('Analyzing Sentiments...')
                    sentiments = get_sentiments(tweets,'English',model_en)
                    
                    st.success('DONE')
                    st.subheader("Results of "+raw_text)
                    
                    #count [Positive - negative - neutral] results
                    counts = pd.Series(sentiments).value_counts()
                    result_df = convert_to_df(counts,'English')
            
                    # display the Dataframe result
                    st.dataframe(result_df)
                    fig1, ax1 = plt.subplots(figsize=(5,5))
                    ax1.pie(result_df['Count'],labels=result_df['Sentiment'].values, autopct='%1.1f%%',
                            shadow=True, startangle=90,colors=['green','yellow','red'])
                    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.    
                    st.pyplot(fig1)

                    #Merging tweets and its sentiments prediction results 
                    tweets_s = pd.Series(tweets,name='Tweet')
                    sentiments_s = pd.Series(sentiments,name='Sentiment (pred)').replace(
                        {'LABEL_2':'Positive','LABEL_1':'Negative','LABEL_0':'Neutral'})
                    all_df = pd.merge(left=tweets_s,right=sentiments_s,left_index=True,right_index=True)
        
        #checking if the user selected the arabic/english model successfully
        if flag==1:
            #reset the flag
            flag=0
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
            
            # display the tweets and sentiments prediction results 
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
            
    #selected the Data Visualization option
    elif choice == "Data Visualization":
        webbrowser.open("https://public.tableau.com/app/profile/marwan.salah5320/viz/SentimentAnalysis_ArabCompanies/Dashboard1?publish=yes")
        del st.session_state['menu_bar']
        st.session_state['menu_bar'] = menu[0]
        choice=menu[0]
        st.experimental_rerun()

    #selected the About option
    elif choice=="About":
        st.subheader("About")
        st.write("This was made in order to have an idea about people's opinion on a certain company. The program scrapes twitter for tweets\
             that are about a certain company. The tweets are then fed into a model for sentiment analysis which is then used \
            to display useful information about each company and public opinion.")
        
        
        st.subheader("This project was developed by the data scientist team during our internship at Digital Factory-Banque Misr .")



if __name__ == '__main__':
	main()