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
We use  [Latent Dirichlet allocation])(https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) which takes a collection of documents and 
discovers the "topics" in each one. Each topic is a distribution over words, it can be interperted as  "How likely is word X when I speak about topic Y". 
 In our story, this allows us to identify the most important words to our script while also comparing them to all of the words in the transcript. 
![alt text](https://github.com/talolard/Interview/blob/master/images/top_n_words.png "The top words for our script")
