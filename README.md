# Enhancing Knowledge Graph Embeddings with Semantic-driven Loss Functions

## Datasets
The ``datasets/`` folder contains the following datasets: ``FB15k187``, ``DBpedia77k``, and ``YAGO14k``. These are the filtered versions of  ``FB15k-237``, ``DBpedia93k``, and ``YAGO19k``, respectively.

The code for generating semantically valid and semantically invalid negative triples is provided for each dataset: ``neg_freebase.py``, ``neg_dbpedia.py``, and ``neg_yago.py``.
These ``.py`` files only need to be run once.
The generated files are: ``sem_hr.pkl`` and ``sem_tr.pkl`` for the semantically valid negative triples; ``dumb_hr.pkl`` and ``dumb_tr.pkl`` for the semantically invalid negative triples.

## Running a model in the command-line
To run a model with vanilla loss functions (the full list of parameters is available in the Usage Section):
Template: `python main_vanilla.py -dataset dataset -model model -batch_size batchsize -lr lr -reg reg -dim dim -lossfunc lossfunc`
Example: `python main_vanilla.py -dataset FB15k187 -model TransE -batch_size 2048 -lr 0.001 -reg 0.001 -dim 200 -lossfunc pairwise` 

To run a model with vanilla loss functions (the full list of parameters is available in the Usage Section):
Template: `python main_sem.py -dataset dataset -model model -batch_size batchsize -lr lr -reg reg -dim dim -lossfunc lossfunc`
Example: `python main_sem.py -dataset FB15k187 -model TransE -batch_size 2048 -lr 0.001 -reg 0.001 -dim 200 -lossfunc pairwise`

Alternatively, one can choose run either the training or testing procedure with the `pipeline` argument:
Template (training): `python main_vanilla.py -pipeline train -dataset dataset -model model -batch_size batchsize -lr lr -reg reg -dim dim -lossfunc lossfunc`
Template (testing): `python main_vanilla.py -pipeline test -dataset dataset -model model -batch_size batchsize -lr lr -reg reg -dim dim -lossfunc lossfunc`

It is also possible to run the ablation study with ``main_vanilla_bucket.py`` and ``main_sem_bucket.py``:
Template: `python main_vanilla_bucket.py -epoch epoch -dataset dataset -model model -batch_size batchsize -lr lr -reg reg -dim dim -lossfunc lossfunc`
where the `epoch` parameter specifies at which epoch to test your model. In our experiments, the `epoch` parameter is set at the best epoch (w.r.t. MRR) found on the validation set.

Details about all the user-defined parameters are available in the Usage Section below.

## Usage

To run your model on a given dataset, the following parameters are to be defined:

`ne`: number of epochs

`lr`: learning rate

`reg`: regularization weight

`dataset`: the dataset to be used

`model`: the knowledge graph embedding model to be used

`dim`: embedding dimension

`batch_size`: batch size

`save_each`: validate every k epochs

`pipeline`: whether training or testing your model from a pre-trained model (or both)

`lossfunc`: the loss function to be used

`monitor_metrics`: whether to keep track of MRR/Hits@/Sem@K during training

`gamma1`: value for gamma1 (pairwise hinge loss)

`gamma2`: value for gamma2 (pairwise hinge loss). This equals $\gamma \cdot \epsilon$ with $\epsilon$ being the semantic factor 

`labelsem`: semantic factor (binary cross-entropy loss)

`alpha`: semantic factor (pointwise logistic loss)

### ConvE

ConvE has additional parameters:

`input_drop`: input dropout

`hidden_drop`: hidden dropout

`feat_drop`: feature dropout

`hidden_size`: hidden size

`embedding_shape1`: first dimension of embeddings

### TuckER

ConvE has additional parameters:

`dim_e`: embedding dimension for entities

`dim_r`: embedding dimension for relations

`input_dropout`: input dropout

`hidden_dropout1`: hidden dropout (first layer)

`hidden_dropout2`: hidden dropout (second layer)

`label_smoothing`: label smoothing