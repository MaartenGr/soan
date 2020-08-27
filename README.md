# SoAn

<p align="center">
<img src="https://github.com/MaartenGr/soan/raw/master/results/calendar.png" height="400"/>
</p>

> Code for applying natural language processing methods on whatsapp conversations

**SoAn** (Social Analysis) can be used to extract word frequency, word clouds, TF-IDF, sentiment analysis, and more
from whatsapp conversations. The main application was initially used to analyze the messages between my wife and me, 
but I extended so that it can be used for your own messages. 


<a name="toc"/></a>
## Table of Contents

1. [Instructions](#instructions)

2. [Output](#output)

    a. [General Plots](#output-general)
    
    b. [TF-IDF](#output-tfidf)
    
    c. [Emoji](#output-emoji)
    
    d. [Sentiment](#output-sentiment)
    
    e. [Word Clouds](#output-wordclouds)
    
    f. [Topic Modeling](#output-topic)
    
<a name="instructions"/></a>
## 1. Instructions
[Back to ToC](#toc)

There are several steps for using this repository:
* **Download** or **fork** this repository
* Install the requirements with `pip install -r requirements.txt`
* Save your whatsapp.txt file in the data folder
  * To download your whatsapp messages simply go open your whatsapp, go to a conversation, click the three vertical dots and export the file
* Finally, from the commandline, run the following:
  * `python soan.py --file whatsapp.txt --language english`
* The results will be saved as images and text files in the results folder

In the notebooks folder, you will also find the **soan.ipynb** where you can run individual pieces of the code. 


<a name="output"/></a>
## 2. Output
[Back to ToC](#toc)


<a name="output-general"/></a>
#### 2.a General Plots

There are 4 types of plots to be generated:
* Messages over time
* Active days of each user
  * Spider
  * Histogram
* Active hours of each user
* Calendar plot

* There are 2 types of stats that are generated:
  * General statistics (text frequency, etc.)
  * Timing
  
Below are some examples of the plots above:

<p align="center">
<img src="https://github.com/MaartenGr/soan/raw/master/results/spider_plot.png" height="400"/>
<img src="https://github.com/MaartenGr/soan/raw/master/results/Me_active_days.png" height="200"/>
<img src="https://github.com/MaartenGr/soan/raw/master/results/moments.png" height="400"/>
</p>

Below are some examples of the text generated:

##########################  
  Number of Messages  
##########################  
  
4444	Her  
3266	Me  
    
#########################  
  Messages per hour    
#########################  
  
Her:	0.1259887165820883  
Me:	0.09259206758710628  
  

<a name="output-tfidf"/></a>
#### 2.b TF-IDF

Using a class-based TF-IDF, I extract the most important words per person and plot them using a horizontal barchart with a mask as image.
I created a horizontal bar chart with two bars stacked on top of each other both plotted on a background image. I started with a background image and plotted the actual values on the left and made it fully transparent with a white border to separate the bars. Then, on top of that I plotted which bars so that the right part of the image would get removed.

<p align="center">
<img src="https://github.com/MaartenGr/soan/raw/master/results/Me_tfidf.png" height="400"/>
<img src="https://github.com/MaartenGr/soan/blob/master/results/Her_tfidf.png" height="400"/>
</p>

**NOTE:** In the notebook, you will see more instructions on how to use your own image. 


<a name="output-emoji"/></a>
#### 2.c Emoji

These analysis are based on the Emojis used in each message. Below you can find the following:

* Unique Emoji per user
* Commonly used Emoji per user

<p align="center">
<img src="https://github.com/MaartenGr/soan/raw/master/results/emoji_Me.png" height="400"/>
</p> 

<a name="output-sentiment"/></a>
#### 2.d Sentiment Analysis

The sentiment from each sentence in the messages is extract per user using Vader and visualized as follows:

<p align="center">
<img src="https://github.com/MaartenGr/soan/raw/master/results/sentiment.png" height="400"/>
</p> 

<a name="output-wordclouds"/></a>
#### 2.e Sentiment Analysis

For each user, a word cloud will be made based on frequent and important words. Stopwords are removed
if you have supplied the language:

<p align="center">
<img src="https://github.com/MaartenGr/soan/raw/master/results/wordcloud_Me.png" height="400"/>
</p> 

<a name="output-topic"/></a>
#### 2.f Topic Modeling

For each user, the most frequent topics using LDA and NMF are modeled and saved a .txt file:

####  
 Me   
####  
  
Topics in nmf model:  
Topic #0: ga boodschappen nodig lieverd halen uurtje half  
Topic #1: thuis wel goed haha lekker we morgen  
Topic #2: lieverd dank hey fijn allerliefste plezier verwacht  
Topic #3: gezellig jeey super jeeeey erg hartstikke samen  
Topic #4: love you most more schattie much very  

**Visualizations Wife**  
Below, you will find an overview of the visualizations I made for my wife, in part using this package:  
<img src="https://github.com/MaartenGr/soan/blob/master/overview.png"/>


