# Natural-Language-Inference

In this repository we are trying to solve [GLUE](https://gluebenchmark.com/) benchmark (mainly [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) dataset) by fine-tuning [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) model.

## MRPC dataset: 
The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.

Split | Count |
--- | --- | 
train | 3,668 |
validation   | 408 |
test  | 1,725 |

Example from the training split:
``` python
{'idx': 58,
 'label': 1,
 'sentence1': 'Several of the questions asked by the audience in the fast-paced forum were new to the candidates .',
 'sentence2': 'Several of the audience questions were new to the candidates as well .'
 }
```

In this notebook [Fine Tune RoBERTa on MRPC Dataset](https://github.com/YamenHabib/Natural-Language-Inference-NLI-/blob/main/Fine%20Tune%20RoBERTa%20on%20MRPC%20Dataset.ipynb) we added two fully connected layers to Roberta model and fine tuned it. we got a 97% acc on the training set and 86.2% on the testing set.

We managed to enhance the accuracy of the model by [pre-training](https://github.com/Alkhaddour/Natural-Language-Inference-NLI-/blob/main/Training%20RoBERTa-based%20model%20using%20STS.ipynb) it on [STS dataset](https://github.com/YamenHabib/Natural-Language-Inference-NLI-/tree/main/stsbenchmark) before [fine-tuning it](https://github.com/Alkhaddour/Natural-Language-Inference-NLI-/blob/main/RoBERTa-based%20model%20trained%20on%20STS%20and%20Fine-Tuned%20on%20MRPC.ipynb) on MRPC. We got a 99.5% acc on the training set and 88.2% on the testing set.
