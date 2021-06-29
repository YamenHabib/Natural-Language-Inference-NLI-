# Natural-Language-Inference

In this repository we are trying to solve [GLUE](https://gluebenchmark.com/) benchmark (mainly [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398) dataset) by fine-tuning [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) model. We also used [STS benchmark](http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark) for pretraining the model and getting better results on the target dataset.

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

## Trained models
All models are based on RoBERTa model. Few layers were added, then they were fine-tuned on one of the above-mentioned datasets. For more details about the models architecture check the [project presentation](https://github.com/YamenHabib/Natural-Language-Inference-NLI-/blob/main/Presention.pptx).

## Models training and evaluation
In order facilitate generation of our results, we provide all necessary details: 
### Requirements
All required libraries are included in [requirements.txt](https://github.com/YamenHabib/Natural-Language-Inference-NLI-/blob/main/requirements.txt) file. A virtual environment could be created using this file (e.g. using Pycharm) to guarantee that there are no conflicts.
### Training the models on STS Benchmark
You can train the model with the default parameters by running:
``` python
python "training RoBERTa-based model on STS benchmark.py" 
```
For more details about parameters run:
``` python
python "training RoBERTa-based model on STS benchmark.py" -h
```

### Training the models on GLEU/MRPC
You can train the model with the default parameters by running:
``` python
python "training RoBERTa-based model on MRPC.py"
```
For more details about parameters run:
``` python
python "training RoBERTa-based model on MRPC.py" -h
```

### Fine tuning the model on MRPC after training it on STS benchmark
In order to increase the model accuracy on MRPC dataset, we trained it first on STS benchmark then contined training on MRPC.
You can train this model with our default parameters by running:
``` python
python "fine-tuning on MRPC after training on STS.py"
```
For more details about parameters run:
``` python
python "fine-tuning on MRPC after training on STS.py" -h
```
<i> NOTE: before running this part, you must have already trained the first model and saved its weights </i>

### Testing model on MRPC
<i>NOTE: Please consider first downloading our trained modelы from [this external link](https://niuitmo-my.sharepoint.com/:f:/g/personal/308544_niuitmo_ru/EjfY5rWkudpIoUdJFMynKI8B2Cl8l6R4D9LY_TBlJGhb1g?e=rPAdDR) or train your own models. In the latter case you might need to change the name of weights file.</i>

To test the model trained on MRPC data only run:
``` python
python "test final model.py" --f_model MRPC_model.pkl --t_model MRPC 
```
To test the model pretrained on STS then fine-tuned on MRPC run:
``` python
python "test final model.py" --f_model MRPC_after_STS_model.pkl --t_model MIXED 
```

## Results:
  <tr>
     <th>     </th>              <th> Training on MRPC only </th> <th>Training on STS then fine-tuning on MRPC</th>
  </tr>
  <tr>
    <td>Train Accuracy</td>      <td> 92.15% </td>                <td> 99.45% </td>
  </tr>
  <tr>
    <td>Validation Accuracy</td> <td> 85.54% </td>                <td> 88.48% </td>
  </tr>
  <tr>
    <td>Test Accuracy</td>       <td> 85.68% </td>                <td> 88.23% </td>
  </tr>
