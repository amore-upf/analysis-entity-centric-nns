
<p align="right">![AMORE-UPF][amore-logo] &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; ![UPF][upf-logo]</p>

# Analysis of Entity-centric Neural Networks
   
Accompanying code for our paper at NAACL-HLT 2019, [What do entity-centric models learn? Insights from entity linking in
multi-party dialogue](URL).  

##### Citation

```
@inproceedings{aina-silberer-sorodoc-westera-boleda:2019:NAACL,
    title     = {What do entity-centric models learn? Insights from entity linking in multi-party dialogue},
    author    = {Aina, Laura and Silberer, Carina and Sorodoc, Ionut-Teodor and Westera, Matthijs and Boleda, Gemma},
    booktitle = {Proceedings of the 2019 Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)},
    month     = {June},
    year      = {2019},
    address   = {Minneapolis, Minnesota},
    publisher = {Association for Computational Linguistics},
}
```

### Contents of this repository
* Code for training, deploying and evaluating different types of entity-centric models and a baseline model on [SemEval 2018 Task 4: Character Identification on Multiparty Dialogues](https://competitions.codalab.org/competitions/17310).
* The trained models referenced in our paper. 
* A dataset we built for probing the entity representations of trained models, which we include in the folder [wikia_task_sentences](/wikia_task_sentences). Note that in the experiment described in our paper we used the sentences of the pattern type 'I' exclusively.
* The script `_fetch_data.sh` for downloading the datasets for [SemEval 2018 Task 4: Character Identification on Multiparty Dialogues](https://competitions.codalab.org/competitions/17310). (Alternatively, you can download the data yourself from [the organizers' github](https://github.com/emorynlp/semeval-2018-task4/tree/master/dat). Store them in the folder data/friends.)

##### A note about PyTorch version

The PyTorch version used here is somewhat old, namely 0.3.0.post4, which can only be installed from a manually downloaded installer (e.g., for python 3.6 and CUDA8.0: https://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl ). For more instructions see [http://pytorch.org/](http://pytorch.org/). 


### Trained models referenced in the paper

The folder [models](models) contains 3 types of models, each in two versions: trained with cross-validation (`_5_folds`, then evaluated as an ensemble) and without (`_1_fold`, a single model). The three model types are, as referenced in our paper:
* `BILSTM`: our baseline model, a plain bidirectional LSTM.
* `ENTLIB`: an implementation of the Entity Library from Aina et al. 2018. 
* `ENTNET`: an implementation of Recurrent Entity Networks from Henaff et al. 2016.


### Usage of main.py for training, deploying and evaluating models
Run the `main.py` script with the corresponding parameters (more details to the different phases are given below): 

`python main.py --phase <phase> [-c <config_file>] [--model <model_path>] [--deploy_data <path_to_data>] [--no_cuda] [--no_eval]`

where 
- `phase` can be train or deploy (optionally runs evaluation)
- `config_file` specifies the hyperparameter settings. Is obligatory for training. 
- `model_path` specifies the path to the model. It is obligatory for the deploy phase. 
- `deploy_data` gives the path to the data for which the model has to output predictions (in [CONLL](https://competitions.codalab.org/competitions/17310#learn_the_details-evaluation) format) (phase: deploy)
- `no_eval` applies to the deploy phase. It can be set if you do not want to evaluate the model, but just want to obtain predictions for some input data. If the input data does not contain target entity ids, `no_eval` is set by default.
- `no_cuda`is set to run the system in CPU mode.


#### Example 1: Deploying and evaluating the ENTLIB model on the SemEval test data

<code>python main.py --deploy_data test --model models/ENTLIB_5_folds/ [--no_cuda]</code>

This will produce the following output files, saved in the directory 
<tt>models/ENTLIB_5_folds/answers/friends_test_scene/</tt> :

- <tt>static_0--ensemble.csv</tt><br/>
*The answer file: It has three columns (called index, prediction, target), <br/>where each row contains the index of the target mention in the test data, the predicted entity id, and the gold entity id to which the mention refers*

- <tt>static_0--ensemble_scores.txt</tt> <br/>
*The evaluation results.*

- <tt>static_0--ensemble_matrix.csv</tt><br/>
*A confusion matrix.*

- <tt>static_0.ini</tt><br/>
*The used config file.*

#### Example 2: Demo training and evaluating from scratch

The demo describes how to train, deploy and evaluate a model from scratch using the official trial data of the SemEval task.

###### Training
`python main.py --phase train -c config_demo.ini [-r] [--no_cuda]`

where the optional parameter 
* `r` is used to activate random sampling of hyperparameters  from intervals specified in the config file. (see <tt>config_demo.ini</tt> for details)
* See above for the description of the other parameters.

The system will produce a subfolder `<year_month>` in the `models` directory, in which it will store several files:
* the config file
* the model file (or files, if run with cross-validation, see parameter `folds` in the config), 
* a `logs` subfolder with the training log (it records the loss, accuracy etc. on the training and validation data for each epoch).

The files will contain a timestamp in their name in the format `<yyyy_mm_dd_hh_mm_ss>`. 

For example, running the command above in May 2019 will train a model with 2-fold cross-validation, and produce something like 
```
.
|__ `models/2019_05/`
|  |  `fixed--2019_05_19_17_58_14.ini`
|  |  `fixed--2019_05_19_17_58_14--fold0.pt`
|  |  `fixed--2019_05_19_17_58_14--fold1.pt`
|  |__`logs/`
|      | `fixed--2019_05_19_17_58_14.log`
|      | `fixed--2019_05_19_17_58_14.ini`
```
The prefix <tt>fixed</tt> means that the model was trained using fixed hyperparameters (since parameter `r` was not set, see above).

###### Using pre-trained word embeddings

Note that the model in this demo initialises the token embeddings randomly. If you want to use  the pre-trained Google News skip-gram word embeddings (as we did for the paper), you first need to download the data from here: 
[GoogleNews-vectors-negative300.bin.gz](https://code.google.com/archive/p/word2vec/).
Put this in the data/ folder. In <tt>config_demo.ini</tt>, set the parameter <tt>token emb</tt> to <tt>google_news</tt>.


###### Evaluation on the trial data
The system was trained using 2-fold cross-validation. So for evaluation on the trial data (on which it was trained), it averages the scores of each fold's models obtained on the respective test split:

`python main.py --phase deploy --deploy_data trial --model models/2019_05/fixed--2019_05_20_11_28_19 [--no_cuda]`

This will produce a subfolder `answers/friends_trial_scene/` in the model subfolder `models/2019_05/`. 
See the [Section above](#running-and-evaluating-the-amore-upf-model-on-the-semeval-test-data) for the description of the files stored therein.

###### Evaluation on the test data
`python main.py --phase deploy --deploy_data test --model models/2019_05/fixed--2019_05_20_11_28_19 [--no_cuda]`

See the [Section above](#running-and-evaluating-the-amore-upf-model-on-the-semeval-test-data) for details.



## Acknowledgements
This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 715154), and from the Spanish Ram\'on y Cajal programme (grant RYC-2015-18907). We are grateful to the NVIDIA Corporation for the donation of GPUs used for this research. We are also very grateful to the Pytorch developers. This paper reflects the authors' view only, and the EU is not responsible for any use that may be made of the information it contains.
<p align="right">![(ERC logo)][erc-logo] &nbsp; ![(EU flag)][eu-flag]</p>


[amore-logo]: https://raw.githubusercontent.com/lauraina/AMORE-semeval/memory-efficient/logos/logo-AMORE-blue-withtext.png?token=AS0KkgGobBv3k09Or_6Bo7AR8r_32Jt9ks5a4u9UwA%3D%3D "A distributional MOdel of Reference to Entities"
[upf-logo]: https://raw.githubusercontent.com/lauraina/AMORE-semeval/memory-efficient/logos/upf-logo.png?token=AS0KkoTyG_CjCMxN8E0CNmXqHUgVoYMOks5a4u9swA%3D%3D "Universitat Pompeu Fabra"
[erc-logo]: https://raw.githubusercontent.com/lauraina/AMORE-semeval/memory-efficient/logos/LOGO-ERC.jpg?token=AS0KksliLXUy3R5G-ri7SBdJJfLDogGcks5a4u5awA%3D%3D "ERC-LOGO"
[eu-flag]: https://raw.githubusercontent.com/lauraina/AMORE-semeval/memory-efficient/logos/flag_yellow_low.jpeg?token=AS0KkpS5zuY5A_fhlrNGwa_9vUYqDHVfks5a4bRpwA%3D%3D "EU-FLAG"

