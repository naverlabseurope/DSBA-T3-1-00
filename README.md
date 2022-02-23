## Due date: March 31, 2022

# Exercise 3
After reading this document ~~[Please look into this notebook](https://colab.research.google.com/drive/1Myb9ttbPggsdgh_fQFRdAt4CK8vpY7X9?usp=sharing)~~
[Please look at this notebook](https://colab.research.google.com/drive/15gP5cn8w7j3Jk_1AuYLB1uyJppx6E1Fn?usp=sharing)

Welcome to your last exercise in this learning journey :slightly_smiling_face: 
We provide you with a notebook containing an example on how to load a pretrained model inside the hugging face library and use it for generation tasks, we would like to ask you to do the same using other models and evaluate on other datasets. 

In particular, in this excercise you will be a master in loading Pretrained models (like T5), writing your own decoding algorithms, as well as investigating what is happening under the hood by interpreting their decisions. You will be (almost) an expert on three tasks Machine Translation, Summarization and Question Answering. 

The notebook contains some examples showing some guiding examples based on BART model. This model is a pretrained one that needs finetuning on the target task to perform well. The cool thing about T5 that it is trained jointly on many tasks both supervised and unsupervised such as LM, translation, summarization and question answering by reforming all tasks as "text" to "text". "For example, automatic summarization is done by feeding in a document
followed by the text “Summarize:” and then the summary is predicted via autoregressive decoding." In this excercise you will have to append those tokens yourself in the input to the model to be able to use it as a summarization model. 

-----

# TASK1: 

## 1.1 Use T5 model on new tasks and dataset
 
### T5 

 ```
 git clone https://github.com/huggingface/transformers.git
 %cd transformers
 pip install .
```
 - You will use `t5-small` pretrained model from Hugging Face: https://huggingface.co/t5-small

 - The documentation of T5 class can be found here: https://huggingface.co/transformers/model_doc/t5.html
 
 To know more about T5 model: 
  - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (https://arxiv.org/pdf/1910.10683.pdf)
  - Stanford guest lecture about T5: http://web.stanford.edu/class/cs224n/slides/cs224n-2021-lecture14-t5.pdf

### New Tasks & Datasets   
- Translation: `bible_para` (https://huggingface.co/datasets/bible_para), `ted_talks_iwslt` (https://huggingface.co/datasets/ted_talks_iwslt) 
- Summarization : `cnn-dailymail` (https://huggingface.co/datasets/cnn_dailymail) 
- Question answering: BoolQ dataset (https://huggingface.co/datasets/boolq)


**Note:** `t5-small` can not handle sequences longer than 512 max_length; you would need to preprocess your datasets accordingly as done above in the tokenizer.

For each task you should give a certain prefix augemented to the input (e.g. "translate English to German: " to be able to translate an english input to german) to know each task prefix consider looking into the config https://huggingface.co/t5-small/blob/main/config.json . 


### Deliverable 1.1  (2 pt)
**OUTPUT:** print some examples from the test/validation split of each task showing the input/model output/ target reference.

## 1.2 - Implement Extra evaluation Metrics

- Summarizaiton: ROUGE (https://www.aclweb.org/anthology/W04-1013.pdf)
- MT: BLEU (https://www.aclweb.org/anthology/P02-1040/)
- Question Answering : exact match and macro-F1 (https://arxiv.org/pdf/1606.05250.pdf) 
- For those you will need a tokenizer you can use an existing implementation of the [MOSES tokenizer](https://pypi.org/project/mosestokenizer/) 
### Deliverable 1.2 (3 pt)
 - implement evaluation metrics : BLEU, ROUGE
 - Select ~1000 sentences from each of the datasets (use `test` when available, or `validation` split otherwise)
 - **Table1**: Evaluate your model on those Metrics
 - **Table2**: As a sanity check of your implementation use already existing implementation online of those metrics and compare them together with your implementation. 

## 1.3- Implement Decoding methods your own
Now you are not allowed to use the existing implementation of the function `model.generate`. [Read here about different usages of this function including many decoding algorithms beam, sampling, top-k and nucleus sampling](https://huggingface.co/blog/how-to-generate)

- **Implement a beam search** generation function that takes beamsize as a function parameter. 
- **Implement a Nucleus sampling** function that samples from a model using Nucleus sampling taking top-p as a function parameter. 
- **Implement Softmax with Temperature** function that samples from a model using Sampling with Temperature taking temperature(t) as a function parameter. 

### Deliverable 1.3 (8pt)

- **Table1:** Check Correctness of your implementation, in a table show a comparison between results obtained from model.generate function and your implementation for different beam-sizes for beamsearch and top-p 

- **Table2:** Compare between different decoding methods. for summarization and machine translation and question answering try different decoding methods for example try changing the top-p value in the nucleus sampling algorithm the temperature of the softmax and with the beamsize in the beam search (for this only you are allowed to use model.generate and existing implementation of evaluation metrics
- **Short Report 300 words max:** Given the results you obtained above. Write a short report containing your conclusions on which are the best decoding algorithm / parameter for each task. Why do you think they are the best? Does increasing the beam size usually give better scores? Why or Why not?

---------------

# Task 2 Attention Visualization 

The goal of this exercise is to understand whether (and how) the attention can be used to interpret model's behaviour. 

## 2.1 Implement cross-attention matrix vizualization
Select several examples for each task and manually examine the attention patterns for each of those tasks. What are your observations? Is there any difference in attention patterns; is there any common patterns? 

### Deliverable 2.1 (3 pt)
**Plots:** You are expected to output plots similar to those in [this blogpost](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
(section attention visualization). 

We expect you to visualize at least three plots showing the following
- Vizualize attention matrices per each head and each layer
- Aggregate the attention values across heads/layers.
- Consider examples from different categories that would take into account: model performance (hard vs easy examples), input length, different task.

**Short report max 300 words:** Add below each of the attention values above. Your comments Highlighting those patterns and what do you observe: eg. common or different patterns across tasks, how those patterns change across layers, individual attention heads versus aggregated attention patterns, any other observations. 

## 2.2 Implement Attention Matrix Aggregation
Manual examination allows to get an intuition of what attention patterns are. Aggregation metrics allow to make corpus-wide conclusions about the roles of different attention heads. Check (this paper)[https://www.aclweb.org/anthology/P19-1580/]  for more details. Implement one or two of the "aggregation" metrics proposed in that paper or (this other paper)[https://aclanthology.org/2021.findings-acl.250/]. Compare the attention patterns across the tasks.  

### Deliverable 2.2  (4 pt) 
- **Plots and short report:** Implement one of these methods for Attention aggregation and plot 3 plots showing some of the aspects above and discuss what do  you learn from aggregated attentions.



----------------------------------


# Bonus 1 (experiment with finetuning)
 - Take any available model on Hugging face which was trained/fine-tuned specifically for the above mentioned tasks (translation, summarization, question answering)
 - Perform task 1 and task 2 with those task-adapted models; Compare it to T5 performance/behaviour.  


## Deliverable Bonus 1 (3pt)
- **Table:** On a single task compare task 1 and 2 using several evaluation metrics and interpretability measures from the above (you can use existing implementation for those metrics). 
- **Short report 300 words max** : comment on What is common and different between these models in terms of interpretability and evaluation metrics? Does the finetuned model perform better than T5 model who was trained on all tasks together? Why would you use one instead of the other? 


# Bonus 2 (implement Minimum Bayes Risk Decoding)
Neural Language Generation models are silly what they believe the highest likely sequence is usually an empty sequence (`<s></s>`) This problem is demonstrated in the following paper: [On NMT Search Errors and Model Errors: Cat Got Your Tongue?](https://www.aclweb.org/anthology/D19-1331/).

This problem is puzzling many scientists at the moment. A method to overcome is to sample many output of the model and rank them according to their pairwise utility. This is a tracktable approximation of a method called Minimum bayes risk decoding. That has been recently proposed in this recent work [Is MAP Decoding All You Need? The Inadequacy of the Mode in Neural Machine Translation](https://arxiv.org/pdf/2005.10283.pdf).
<img src=https://i.imgur.com/J0ePay7.png width=500>

In this bonus task we ask you to implement this decoding method as the two one above (you can use any utility function of your choice in the paper they use METEOR python implementation is available online e.g. here https://pypi.org/project/textmetrics/).

## Deliverable Bonus 2 (5pt):
- **Table:** Compare MBR decoding vs Beam search with beam size=5, beam size=10, beam size=15 other on machine translation task above.  
- **Short Report 300 words max:** Given the results you obtained above. Write a short report containing your conclusions. What on which are the best decoding algorithm / parameter for each task. Why is that? what are your conclusions?  

------------------

# Summary of All Deliverables
Overall you have Two tasks with 8 deliverables with 3 optional ones: 
- Deliverable 1.1  (2 pt) 
- Deliverable 1.2 (3 pt)
- Deliverable 1.3 (8 pt)
- Deliverable 2.1 (3 pt)
- Deliverable 2.2  (4 pt) 
- Deliverable Bonus 1 (3pt)
- Deliverable Bonus 2 (5pt)

# Submit your Exercise

- All deliverables are expected to be submitted in a single colab notebook. 
- In your notebook please highlight each deliverable by its title (e.g. # Deliverable 1.2)..etc
- Please stick to the format of each deliverable being a table short report or a plot as identified above 
- Please name your notebook  on the following format  DSBA_EXCERCISE3_FIRSTNAME_LASTNAME
(where firstname and lastname are those of the one who will submit the exercise on behalf of the team)
- Please make sure that your notebook is publicly accessible through the provided URL. 

Submit your excercise by filling the following form (one submission per team): https://forms.gle/nqkcUw3v6oLxHEQJ6
