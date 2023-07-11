## Selected projects in data science, machine learning, statistics and NLP

---


### NLP - Pre-train Longformer with masked language modelling on german party speeches


- In a previous script, the longformer was trained on the German language and the type of language used in political speeches of the German parliament. 

- Now the fine-tuning takes place with the corresponding labels of the parties. 


- With the help of the Longformer it is possible to process an input of 4096 tokens, which breaks the limitations of other Bert models. 

-  The training takes a very long time and requires a GPU with a lot of RAM. So far, 45 epochs have been trained, each requiring 1.5 hours. You can examine the results so far under "6. test".


<img src="images/Longformer.png?raw=true"/>


[View on google colab](https://colab.research.google.com/drive/1dfgr4Wbd9Pp-vh1nvWShdO0KL8cYYHcY?usp=sharing/)


---
---

### NLP - Pre-train Longformer with masked language modelling on german party speeches


- For long texts the Longformer is a suitable choice, because it can process 4096 inputs of tokens. 
The goal: The Longformer should be trained with speeches of politicians from the Bundestag, so that the 
German language and the peculiarity of the language of the speeches are learned.

- Since the Longformer is based on RoBerta, the ByteLevel Tokenizer or RoBerta Tokenizer is used. 

- 15% of the texts are masked and completely given into the model. 

- The model was trained with about 35 epochs, a batch size of 4 and a learning rate of 0.00001 and 0.000001. 
A higher batch size leads to an overload of the working memory of the GPU.

- One can see that the word 'colleague' is correctly predicted with almost 100%. Some words are not predicted correctly, 
but the sense and grammar is correct. 

- The model was trained on a A100 80G GPU.

So,the language model works!


<img src="images/Longformer.png?raw=true"/>


[View on google colab](https://colab.research.google.com/drive/1q0vrNHLWiyP3GUxb0Yv8wfzHmTyMjR4E?usp=sharing/)

---

---

### NLP - Classification with tf-idf and interpretation with lime 

In this script, I search for the best model for prediction and configure the lime explainer for texts

- The data was collected in-house and a sample of the party speeches was created.
- The data were cleaned.
- Calculate tf-idf features
- Check out many models for the best prediction
- Check flase classifications
- Interprete the predictions with lime explainer


<img src="images/text_lime.png?raw=true"/>


[View on google colab](https://colab.research.google.com/drive/1OscdJQUFy9wokk653kCae02xTSxWN3aa?usp=sharing/)




---

---

### NLP - Doc2Vec, Evaluation, Classification. Also Classification with CNN.

In this script, I classified with the help of Doc2Vec. A possibility to vectorise documents of any length. 

- The data was collected in-house and a sample of the party speeches was created.
- The data were cleaned.
- Doc2Vec was made possible with the help of Gensim. Both DBOW and DM were calculated.
- Evaluation of Doc2Vec: The model was good when the respective document was most similar to itself.
- Classification with logistic Regression, tree based algorithms and all kinds of SVMs.
- Classification with CNN.
- The results are worse than those with TF-IDF


[View on google colab](https://colab.research.google.com/drive/1dTdfws5Vsy8oudq4ItiAjikiG_4iR6Rt?hl=de/)


---

---


### NLP - LDA Topic-Modeling with party speeches of the Bundestag of WP19

In this script, topics were modeld with the Mallet version of LDA

- The data was collected in-house and a sample of the party speeches was created.
- The data were cleaned.
- The best coherence value was looked for.
- The most dominant theme per speech was selected.
- The proportion of each topic per session was calculated.
- 3 topics were selected. So one can see its progress.
- Plotting with pygal.


<img src="images/LDA_Topics.png?raw=true"/>

[Click here!](https://jako1.w3spaces.com){:target="_blank"}

[View on google colab](https://colab.research.google.com/drive/1r2jQIj2lu4Rgf5qpwEzIYbJ0ksAPiX-m?usp=sharing/)


---


---


### NLP - BERT finetuning with huggingface german BERT model for classification

In this script, I present a simple workflow for finetuning a pretrained BERT model

- The data was collected in-house and a sample of the party speeches was created.
- The data were cleaned.
- The data were prepared for the huggingface model finetuning
- Simple NN for finetuning party speeches
- Prediction of propabilities: When someone enters a text, you can see how likely this 
  text corresponds to the respective parties.
- Visualisation und interaction with gradio

<img src="images/gradio BERT.png?raw=true"/>

[View on google colab](https://colab.research.google.com/drive/1WU9ZzQDJ-pwHelfyzXln8MJ22wfHKA6k?usp=sharing/)





---


---

### NLP - Word2Vec with party speeches of the Bundestag of WP19

In this script, word vectors are created using your own data.

- The data was collected in-house and a sample of the party speeches was created.
- The data were cleaned
- A neural network was created (not from gensim).
- Similar vectors were selected based on keywords. Procedure: SkipGram
- The data points of the resulting words were reduced to 2-dimensions using tsne.
- The data points were plotted with Bokeh.
- You can clearly see the similarities because the themes were coloured in.

- Result: The procedure works.
- But: No statements can be made about the parties themselves. Differences in context are not reflected in the words.


<img src="images/bokeh_plot-2.png?raw=true"/>

[Click here!](https://www.janoschkorell.eu/wp-content/uploads/2023/05/party_speech_tsne.html){:target="_blank"}

[View on google colab](https://colab.research.google.com/drive/1tjqPQcvTm8ZkJBp1-Zlw4ZEa3EqlYOCU?usp=sharing/)


---


---

### App/NLP - Predict text with text from party speeches

This app was created with dash and plotly so that interactions and inputs are possible. With it, it is possible to predict text with LSTM models trained with speeches of politicians from the German Bundestag. This is part of a larger project: "Bundestag in data". You can ask a question to three models at the same time and get different answers from different parties.


<img src="images/App - PTWT.png?raw=true"/>

[View on google colab](https://colab.research.google.com/drive/1B47jgrPcoOJAS6UaKu6xzuYJCktM_5fw?usp=sharing/)


---


---



### Textmining - Scraping individual politicians speeches


Script to scrape the speeches in order of time, in order and by speaker from the plenary transcripts.
I made this with regex strategies.

<img src="images/Plenarprotokoll.png?raw=true"/>

[View on github](https://github.com/JanoschKorell/Text-Mining---scraping-individual-politicians-speeches-from-texts)


---


---


### Machine Learning/NLP - Use LSTMs to learn rules of speeches


With this script, LSTMs were used to learn the rules of politicians' speeches from the Bundestag and their partisan context. This way, text can be generated through queries that reflect the nature of the respective party. 


[View on google colab](https://colab.research.google.com/drive/14wrXcbDBefyQZRu2fDnFkqbJefkeUdL5?usp=sharing)


---



---


### Machine Learning - Predict bad cars and interpreting (explainable ai)


This project is a complete Datascience workflow consisting of:

EDA, Outliers, Imputation, Train-Test-Split, Dim Reduction, Feature Engineering, Hyperparameter Optimization, Train Model (XGBoost), Classification report, Interpretation with SHAP Values.


<img src="images/shap values.png?raw=true"/>


[View on github](https://github.com/JanoschKorell/Predict-bad-cars---Full-Data-Sience-Project)


---


---


### Research project - Determinants of the perception of justice in Europe


Logistic multilevel regression with Stata; Descriptive statistics, level 1 and level 2 hypotheses.

This work is in the field of empirical justice research and aimed to explain determinants of perceptions of justice in Europe. Due to data protection reasons, only the STATA code can be published.


<img src="images/SRC.png?raw=true"/>


[View on github](https://github.com/JanoschKorell/Researchproject--Determinants-of-the-Perception-of-Justice-in-Europe)



---


---


### Web scraping - Twitter scraper without API


Because the previous ways to do data mining on Twitter are limited to sometimes poorly functioning libraries using the Twitter API, I wrote a piece of code over which I have my own control and for which no API is necessary.


[View on github](https://github.com/JanoschKorell/Twitter-Scraper-without-API-)






---


---


### Plots - STATA Plots


This repository is to show my understanding of how data is represented. The plots have been chosen according to the structure of the data. For the programming of the plots STATA was used. You can view these files to also see my skills in programming STATA.

[View on github](https://github.com/JanoschKorell/Various-plots)


<img src="images/collage.jpg?raw=true"/>

---
<p style="font-size:11px">Page template forked from <a href="https://github.com/evanca/quick-portfolio">evanca</a></p>
<!-- Remove above link if you don't want to attibute -->
