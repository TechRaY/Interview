# Interview Solution
This is a solution to a technical challange a company gave me.
A solution to a technical challenge I got from a company. The task is given some data:
1. Compute the most important key-words (a key-word can be between 1-3 words)
2. Choose the top n words from the previously generated list. Compare these key-
words with all the words occurring in all of the transcripts.
3. Generate a score (rank) for these top n words based on analysed transcripts.

## Data
The data is 4 wikipedia articles each relating to food.
**script.txt** is an article about food

**transcript_1.txt** is about Fast Food

**transcript_2.txt** is about a resteraunt

**transcript_3.txt** is about Cooking
## My interpertation

To figure out what is "important" I need to build a story explaining what kind of value this task will drive.
Seeing as the files are called *transript_x* and *script* I'll invent the following story:

> We are a call center selling widgets. Our sales people have a script that they follow which reflects our company and product.
> We want to help our sales people engage with clients, by identifying the terms that they should use more in their calls. With a few contraints 
> 1. We want the terms to be as relevant to our brand and message as possible
> 2. We want the terms our customers use frequently and we don't. 

## My solution 
To find the words that are most relevant to the brand we'll compare our script to call transcripts as well as to some other documents. 
We use  [Latent Dirichlet allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) which takes a collection of documents and 
discovers the "topics" in each one. 
### Getting our top n words and comparing to the full corpus
Each topic is a distribution over words, it can be interperted as  "How likely is word X when I speak about topic Y". 
 In our story, this allows us to identify the most important words to our script while also comparing them to all of the words in the transcript. 

The 20 most significant words in our script are 
>	food 	restaurant 	fast
	**fast food** 	wine	world
	country	people	year
	culture	price	known
	common	market	meal
	include	service	type
	local	served
>

Each of the documents we have is composed of a mixture of topics. The following chart shows how each document is "composed". Crucially, we have a distinct topic
for our script and that is where the top words come from 

![Topic distribution](https://github.com/talolard/Interview/blob/master/images/topics.png "Topic breakdown by document")

### Ranking our words : Finding words our sales staff should use more. 

To find words that were relevant to the script but were not used enough in calls, I trained a TFIDF "vectorizer" on the transcripts only.
 Applying it to our script generates a tfidf score per word, which is simply how frequently that word appears in our script divided by how
 frequently it appears in the transcripts. Since the vectorizer was trained only on the transcripts, words that don't appear in the transcripts get 
   a null score. Taking the negative log of the tfidf gives us a smoothed score. The score is high when the word is prominent in the transcripts but 
    infrequent in the script, showing us which words are relevant to cusomters but are not frequently used. By ranking the top n words in our script
    topic with this score we obtain a ranking of the words that are most important to us that we should use more. 
    
  
![Words to use more](https://github.com/talolard/Interview/blob/master/images/top_n_words.png "Words we should use more")

##Tricks and notes

1. I trained the LDA and Vectorizer on individual sentances instead of the entire document. This increased the "document frequency" wheight of 
each word which was important given the small number of documents
2. The help the LDA model seperate topics I introduced extra documents to the training corpus. I added documents about 
Algebraic topology and the Banch Tarsky paradox.
3. To learn finer grained topics in our domain I added documents about Pizza and Wine. 

# Using this

## Installation. 
Clone the repository

```` 
pip install -r requirements.txt
````

You may need to download NLTK corpuses. Open python in the terminal and run 

````
import nltk
mltk.download()
````

To run with default settings
```
python main.py
```

## Adding a new corpus
You can train on any script and transcriptions. To use a new data set: 
1. Create a new directory where you wish to store the data
2. Add a single script.txt file
3. Add as many transcript_{}.txt files as you wish
4. Optionally, add other .txt files 

Then run
 ```
python main.py --data_dir=my_data_dir
```

