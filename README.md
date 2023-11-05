# Treat Different Negatives Differently: Enriching Loss Functions with Domain and Range Constraints for Link Prediction

## Datasets
The ``datasets/`` folder contains the following datasets: ``FB15k187``, ``DBpedia77k``, and ``YAGO14k``. These are the filtered versions of  ``FB15k-237``, ``DBpedia93k``, and ``YAGO19k``, respectively [1].

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

`python main_vanilla_bucket.py -epoch epoch -dataset dataset -model model -batch_size batchsize -lr lr -reg reg -dim dim -lossfunc lossfunc`

`python main_sem_bucket.py -epoch epoch -dataset dataset -model model -batch_size batchsize -lr lr -reg reg -dim dim -lossfunc lossfunc`

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

## Full hyperparameter space

All models were tested with the following combinations of hyperparameters:

| Hyperparameters                        | Range                                           |
|---------------------------------------|-------------------------------------------------|
| Batch Size                            | {128, 256, 512, 1024, 2048}                   |
| Embedding Dimension                   | {50, 100, 150, 200}                           |
| Regularizer Type                      | {None, L1, L2}                                |
| Regularizer Weight ($\lambda$)        | {1e-2, 1e-3, 1e-4, 1e-5}                     |
| Learning Rate ($lr$)                  | {1e-2, 5e-3, 1e-3, 5e-4, 1e-4}              |
| Margin $\gamma~(\mathcal{L}_{PHL})$    | {1, 2, 3, 5, 10, 20}                          |
| Semantic Factor $\epsilon~(\mathcal{L}^{S}_{PHL})$ | {0.01, 0.1, 0.25, 0.5, 0.75}         |
| Semantic Factor $\epsilon~(\mathcal{L}^{S}_{PLL})$ | {0.05, 0.10, 0.15, 0.25}             |
| Semantic Factor $\epsilon~(\mathcal{L}^{S}_{BCEL})$ | {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}  |


## Best hyperparameters found

| Model      | Hyperparameters         | DBpedia77k | FB15k187 | Yago14k |
|------------|-------------------------|------------|----------|---------|
| TransE     | Batch Size              | 2048       | 2048     | 1024    |
|            | Embedding Dimension     | 200        | 200      | 100     |
|            | Learning Rate           | 0.001      | 0.001    | 0.001   |
|            | Regularization Weight   | 0.001      | 0.001    | 0.001   |
|            | Semantic Factor         | 0.5        | 0.25     | 0.25    |
| TransH     | Batch Size              | 2048       | 2048     | 1024    |
|            | Embedding Dimension     | 200        | 200      | 100     |
|            | Learning Rate           | 0.001      | 0.001    | 0.001   |
|            | Regularization Weight   | 0.00001    | 0.00001  | 0.00001 |
|            | Semantic Factor         | 0.5        | 0.25     | 0.25    |
| DistMult   | Batch Size              | 2048       | 2048     | 1024    |
|            | Embedding Dimension     | 200        | 200      | 100     |
|            | Learning Rate           | 0.1        | 10.0     | 0.0001  |
|            | Regularization Weight   | 0.00001    | 0.00001  | 0.00001 |
|            | Semantic Factor         | 0.5        | 0.25     | 0.25    |
| ComplEx    | Batch Size              | 2048       | 2048     | 1024    |
|            | Embedding Dimension     | 200        | 200      | 100     |
|            | Learning Rate           | 0.001      | 0.001    | 0.01    |
|            | Regularization Weight   | 0.1        | 0.1      | 0.1     |
|            | Semantic Factor         | 0.15       | 0.15     | 0.015   |
| SimplE     | Batch Size              | 2048       | 2048     | 1024    |
|            | Embedding Dimension     | 200        | 200      | 100     |
|            | Learning Rate           | 0.1        | 0.1      | 0.1     |
|            | Regularization Weight   | 0.01       | 0.1      | 0.00001 |
|            | Semantic Factor         | 0.15       | 0.15     | 0.15    |
| ConvE      | Batch Size              | 512        | 128      | 512     |
|            | Embedding Dimension     | 200        | 200      | 200     |
|            | Learning Rate           | 0.001      | 0.001    | 0.001   |
|            | Regularization Weight   | 0.0        | 0.0      | 0.0     |
|            | Semantic Factor         | 0.0001     | 0.001    | 0.001   |
| TuckER     | Batch Size              | 128        | 128      | 128     |
|            | Embedding Dimension     | 200        | 200      | 100     |
|            | Learning Rate           | 0.001      | 0.0005   | 0.001   |
|            | Regularization Weight   | 0.0        | 0.0      | 0.0     |
|            | Semantic Factor         | 0.00001    | 0.0001   | 0.0001  |
| RGCN       | Embedding Dimension     | 500        | 500      | 500     |
|            | Learning Rate           | 0.01       | 0.01     | 0.01    |
|            | Regularization Weight   | 0.01       | 0.01     | 0.01    |
|            | Semantic Factor         | 0.1        | 0.1      | 0.1     |


## References
[1] Hubert, N., Monnin, P., Brun, A., & Monticolo, D. (2023). [Sem@K: Is my knowledge graph embedding model semantic-aware?](https://arxiv.org/abs/2301.05601)
