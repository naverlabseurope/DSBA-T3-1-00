# Goal
The goal of this exercise is to analyse T5 model performance from different aspects
 - inference
 - interpertability

# T5 model
 - For this exercice you would need to get last dev version (`4.4.0.dev0`) of transformers from github
 
 ```
 git clone https://github.com/huggingface/transformers.git
 %cd transformers
 pip install .
```
 - You will use `t5-small` model from Hugging Face: https://huggingface.co/t5-small
 
 To know more about T5 model: 
  - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (https://arxiv.org/pdf/1910.10683.pdf)
  - Stanford guest lecture about T5: http://web.stanford.edu/class/cs224n/slides/cs224n-2021-lecture14-t5.pdf
  
## Datasets
- translation: `bible_para` (https://huggingface.co/datasets/bible_para), `ted_talks_iwslt` (https://huggingface.co/datasets/ted_talks_iwslt) 
- summarization : `cnn-dailymail` 
- question answering: `squad`

Note: `t5-small` can not handle sequences longer than 512 tokens; you would need to preprocess your datasets accordingly. Select ~1000 sentences from each of the datasets (use `test` when available, or `validation` split otherwise) 

## Evaluations
- Summarizaiton: ROUGE (https://www.aclweb.org/anthology/W04-1013.pdf)
- MT: BLEU (https://www.aclweb.org/anthology/P02-1040/)
- Question Answering : exact match and macro-F1 (https://arxiv.org/pdf/1606.05250.pdf) 
- evaluate the metrics or use existing implementations?
 
# Task 1 Evaluation metrics (3 pt)
 - implement evaluation metrics : BLEU, ROUGE
 - Optional: implement additional evaluation metrics for translation and/or summarization tasks (bonus: 3 pt)
 
# Task 1 Inference (8 pt)
link to (notebook1)
- Implement Softmax with Temp / Nucleus / Beam Search
- T5 take all the tasks write a report on how different decoding algorithms with T5
 
## Questions

# Task 2 Attention Visualization (8 pt) 

The goal of this exercice is to understand whether (and how) the attention can be used to interpret model's behaviour. 
1.  (4 pt) Implement cross-attention matrix vizualization
  -  select several examples for each task and manually examine the attention patterns for each of those tasks. What are your observations? Is there any difference in attention patterns; is there any common patterns? You can vizualize attention matrices per each head and each layer, or try to aggregate the attention values across heads/layers.
  -  Consider examples from different categories that would take into account: model performance, input length, other?   Can you observe any patterns depending on the nature of example?
    
2.  (4 pt) Manual examination allows to get an intuition of what attention patterns are. Aggregation metrics allow to make corpus-wide conclusions about the roles of different attention heads. Check (this paper)[https://www.aclweb.org/anthology/P19-1580/]  for more details. Implement one of the "aggregation" metrics proposed in that paper : confidence score or LRP. Compare the attention patterns across the tasks.  
 
 
# Task 3 (bonus? - 5pt)
 - take any available model on Hugging face which was trained/fine-tuned specifically for the above mentioned tasks (translation, summarization, question answerint)
 - perform task 1 and task 2 with those task-adapted models; Compare it to T5 performance/behaviour, and comment what is in common, and what is different. Think of potential reasons for such behaviour.  

# Deliverables


## Task 1 
- Report (Answering Questions) Tables of comparison 
- Link to a colab notebook


## T5 tasks 
Present a short summary on T5  (Answer some questions)
