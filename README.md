# LightCTR

LightCTR is a tensorflow 2.0 based, extensible toolbox for building CTR/CVR predicting models. 
It provides several off-the-shelf popular CTR models for you to use. And it also contains some useful
model blocks to help you build your own model quickly.

Have a quick start with the example script `./examples/ctr_predict.py`

A small example dataset `avazu_1w.txt` is provided in `./datasets`, It is sampled from a [kaggle
dataset](https://www.kaggle.com/c/avazu-ctr-prediction) with 10000 rows.

## Models List

|                    Model                     | Paper   |
| :------------------------------------------: | :---------------- |
| Factorization-supported Neural Network (FNN) | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)|
|      Product-based Neural Network (PNN)      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)|
|                 Wide & Deep                  | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)|
|                    DeepFM                    | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)|
|          Deep & Cross Network (DCN)          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|                   xDeepFM                    | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|      Neural Factorization Machine (NFM)      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|   Attentional Factorization Machine (AFM)    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|                   AutoInt                    | [arxiv 2018][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |
|  Convolutional Click Prediction Model (CCPM) | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)                    |
| Feature Generation by Convolutional Neural Network (FGCNN) | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)   |
|       Mixed Logistic Regression (MLR)        | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/pdf/1704.05194.pdf) |
|                     FiBiNET                  | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |
|                     NFFM                     | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                           |
