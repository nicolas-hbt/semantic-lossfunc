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


## Chosen hyperparamters (main experiment)

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


## Appendices

This section aims at providing implementation details that could not be discussed in the paper's content due to page limitations.

### Chosen hyperparameters for the KGEMs trained with the alternative semantic-driven loss function

| Model   | Hyperparameters         | DBpedia77k | FB15k187 | Yago14k |
|---------|-------------------------|------------|----------|---------|
| ComplEx | Batch Size              | 2048       | 2048     | 1024    |
|         | Embedding Dimension     | 200        | 200      | 100     |
|         | Learning Rate           | 1e-4       | 1e-4     | 1e-3    |
|         | Regularization Weight   | 1e-1       | 1e-1     | 1e-1    |
|         | Semantic Factor         | -1e-1      | -1e-1    | 1e-2    |
| SimplE  | Batch Size              | 2048       | 2048     | 1024    |
|         | Embedding Dimension     | 200        | 200      | 100     |
|         | Learning Rate           | 1e-3       | 1e-4     | 1e-3    |
|         | Regularization Weight   | 1e-1       | 1e-1     | 1e-1    |
|         | Semantic Factor         | -1e-1      | 1e-2     | 1e-2    |
| ConvE   | Batch Size              | 512        | 128      | 512     |
|         | Embedding Dimension     | 200        | 200      | 200     |
|         | Learning Rate           | 1e-3       | 1e-3     | 1e-3    |
|         | Regularization Weight   | 0          | 0        | 0       |
|         | Semantic Factor         | 1e-6       | 1e-5     | 1e-4    |
| TuckER  | Batch Size              | 128        | 128      | 128     |
|         | Embedding Dimension     | 200        | 200      | 100     |
|         | Learning Rate           | 1e-3       | 5e-4     | 1e-3    |
|         | Regularization Weight   | 0          | 0        | 0       |
|         | Semantic Factor         | 1e-6       | 1e-5     | 1e-5    |
| RGCN    | Embedding Dimension     | 500        | 500      | 500     |
|         | Learning Rate           | 1e-2       | 1e-2     | 1e-2    |
|         | Regularization Weight   | 1e-2       | 1e-2     | 1e-2    |
|         | Semantic Factor         | 1e-4       | 1e-5     | 1e-4    |


### Cut-offs for FB15k187, DBpedia77k, and Yago14k

Cut-offs for FB15k187, DBpedia77k, and Yago14k. B1, B2, and B3 denote the buckets of relations with narrow, intermediate, and large sets of semantically valid heads or tails, respectively. $|\mathcal{R}|$ denotes the number of unique relations in a given bucket and $|\text{Sem. Val}|$ indicates the interval of the number of semantically valid entities for the bucket relations. To illustrate, $|\text{Sem. Val}|$ = [11, 216] for the head side means that relations in the bucket have at least $11$ and at most $216$ semantically valid heads.

| Bucket | Side | Sem. Val Range | Unique Relations | Sem. Val Range | Unique Relations | Sem. Val Range | Unique Relations |
|--------|------|----------------|------------------|----------------|------------------|----------------|------------------|
|        |      | FB15k187        |                  | DBpedia77k     |                  | Yago14k        |                  |
|        |      | Sem. Val Range  | Unique Relations | Sem. Val Range | Unique Relations | Sem. Val Range | Unique Relations |
|--------|------|----------------|------------------|----------------|------------------|----------------|------------------|
| B1     | Head | [11, 216]      | 69               | [12, 930]      | 62               | [93, 811]      | 10               |
|        | Tail | [12, 244]      | 80               | [19, 801]      | 44               | [35, 678]      | 13               |
| B2     | Head | [278, 1391]    | 55               | [1295, 11586]  | 58               | [2102, 3624]   | 15               |
|        | Tail | [278, 1391]    | 49               | [1419, 11586]  | 55               | [2102, 3624]   | 16               |
| B3     | Head | [1473, 4500]   | 63               | [22252, 57242] | 25               | {5730}         | 12               |
|        | Tail | [1473, 4500]   | 58               | {57242}        | 50               | {5730}         | 8                |

### Rank-based and Semantic-based Results on DBpedia77k (Intermediate and Large Sets)


Rank-based and semantic-based results on DBpedia77k for buckets of relations that feature an intermediate (B2) and large (B3) set of semantically valid heads or tails.
| Model        | MRR   | H@10  | S@10  | MRR   | H@10  | S@10  |
|--------------|-------|-------|-------|-------|-------|-------|
|              | B2    | B2    | B2    | B3    | B3    | B3    |
|              | MRR   | H@10  | S@10  | MRR   | H@10  | S@10  |
|--------------|-------|-------|-------|-------|-------|-------|
| TransE-V     | .450  | .607  | .838  | .317  | .429  | .995  |
| TransE-S     | .404  | .556  | .987  | .300  | .407  | 1     |
|--------------|-------|-------|-------|-------|-------|-------|
| TransH-V     | .449  | .610  | .729  | .311  | .425  | .971  |
| TransH-S     | .423  | .592  | .981  | .296  | .413  | 1     |
|--------------|-------|-------|-------|-------|-------|-------|
| DistMult-V   | .446  | .553  | .669  | .505  | .413  | .742  |
| DistMult-S   | .450  | .566  | .790  | .506  | .422  | .920  |
|--------------|-------|-------|-------|-------|-------|-------|
| ComplEx-V    | .442  | .538  | .551  | .582  | .453  | .787  |
| ComplEx-S    | .448  | .545  | .707  | .505  | .426  | .975  |
|--------------|-------|-------|-------|-------|-------|-------|
| SimplE-V     | .381  | .461  | .716  | .485  | .357  | .954  |
| SimplE-S     | .350  | .404  | .649  | .386  | .276  | .960  |
|--------------|-------|-------|-------|-------|-------|-------|
| ConvE-V      | .388  | .535  | .890  | .489  | .371  | .960  |
| ConvE-S      | .429  | .559  | .977  | .450  | .399  | .999  |
|--------------|-------|-------|-------|-------|-------|-------|
| TuckER-V     | .438  | .547  | .874  | .591  | .436  | .898  |
| TuckER-S     | .444  | .568  | .923  | .564  | .444  | .983  |
|--------------|-------|-------|-------|-------|-------|-------|
| RGCN-V       | .282  | .413  | .670  | .367  | .322  | .971  |
| RGCN-S       | .275  | .423  | .861  | .362  | .357  | .999  |

### Rank-based and Semantic-based Results on FB15k-187 (Intermediate and Large Sets)

Rank-based and semantic-based results on FB15k187 for the buckets of relations that feature an intermediate (B2) and large (B3) set of semantically valid heads or tails.
| Model        | MRR   | H@10  | S@10  | MRR   | H@10  | S@10  |
|--------------|-------|-------|-------|-------|-------|-------|
|              | B2    | B2    | B2    | B3    | B3    | B3    |
|              | MRR   | H@10  | S@10  | MRR   | H@10  | S@10  |
|--------------|-------|-------|-------|-------|-------|-------|
| TransE-V     | .330  | .526  | .934  | .141  | .255  | .953  |
| TransE-S     | .385  | .588  | .972  | .169  | .290  | .993  |
|--------------|-------|-------|-------|-------|-------|-------|
| TransH-V     | .330  | .517  | .846  | .161  | .262  | .963  |
| TransH-S     | .380  | .590  | .967  | .171  | .291  | .993  |
|--------------|-------|-------|-------|-------|-------|-------|
| DistMult-V   | .336  | .527  | .780  | .177  | .274  | .946  |
| DistMult-S   | .388  | .579  | .962  | .187  | .309  | .995  |
|--------------|-------|-------|-------|-------|-------|-------|
| ComplEx-V    | .327  | .476  | .318  | .197  | .306  | .717  |
| ComplEx-S    | .351  | .537  | .769  | .191  | .310  | .942  |
|--------------|-------|-------|-------|-------|-------|-------|
| SimplE-V     | .283  | .432  | .331  | .179  | .274  | .694  |
| SimplE-S     | .283  | .448  | .671  | .159  | .243  | .923  |
|--------------|-------|-------|-------|-------|-------|-------|
| ConvE-V      | .347  | .529  | .974  | .172  | .277  | .977  |
| ConvE-S      | .354  | .543  | .998  | .188  | .283  | .999  |
|--------------|-------|-------|-------|-------|-------|-------|
| TuckER-V     | .387  | .574  | .987  | .215  | .330  | .994  |
| TuckER-S     | .396  | .585  | .991  | .222  | .337  | .997  |

### Rank-based and Semantic-based Results on Yago14k (Intermediate and Large Sets)

Rank-based and semantic-based results on Yago14k for the buckets of relations that feature an intermediate (B2) and large (B3) set of semantically valid heads or tails.
| Model        | MRR   | H@10  | S@10  | MRR   | H@10  | S@10  |
|--------------|-------|-------|-------|-------|-------|-------|
|              | B2    | B2    | B2    | B3    | B3    | B3    |
|              | MRR   | H@10  | S@10  | MRR   | H@10  | S@10  |
|--------------|-------|-------|-------|-------|-------|-------|
| TransE-V     | .879  | .928  | .892  | .841  | .923  | .974  |
| TransE-S     | .861  | .922  | .997  | .854  | .917  | 1     |
|--------------|-------|-------|-------|-------|-------|-------|
| TransH-V     | .854  | .922  | .567  | .788  | .92   | .803  |
| TransH-S     | .865  | .921  | .876  | .778  | .926  | .996  |
|--------------|-------|-------|-------|-------|-------|-------|
| DistMult-V   | .852  | .915  | .443  | .941  | .911  | .536  |
| DistMult-S   | .862  | .911  | .441  | .941  | .911  | .584  |
|--------------|-------|-------|-------|-------|-------|-------|
| ComplEx-V    | .883  | .921  | .352  | .932  | .914  | .619  |
| ComplEx-S    | .881  | .918  | .738  | .922  | .914  | .964  |
|--------------|-------|-------|-------|-------|-------|-------|
| SimplE-V     | .882  | .915  | .378  | .932  | .914  | .656  |
| SimplE-S     | .883  | .918  | .841  | .930  | .905  | .991  |
|--------------|-------|-------|-------|-------|-------|-------|
| ConvE-V      | .893  | .928  | .858  | .941  | .917  | .904  |
| ConvE-S      | .892  | .925  | .931  | .939  | .923  | .956  |
|--------------|-------|-------|-------|-------|-------|-------|
| TuckER-V     | .884  | .928  | .791  | .941  | .917  | .915  |
| TuckER-S     | .894  | .935  | .930  | .942  | .917  | .983  |





## References
[1] Hubert, N., Monnin, P., Brun, A., & Monticolo, D. (2023). [Sem@K: Is my knowledge graph embedding model semantic-aware?](https://arxiv.org/abs/2301.05601)
