# Text Classification using Multi Layer Perceptron Neural Network with Word Embeddings 

This repository implements MLP with word embeddings for text classification using PyTorch.

To run the repository, ensure that PyTorch is first installed. Instructions on PyTorch installing can be found in the official website https://pytorch.org/.

PyTorch 1.7.1 was used during the experiments but the code should work with different versions of PyTorch.

To prevent overfitting, the final model uses just a single linear layer with size = 100 and embedding = 50. Average of the embeddings is taken to reduce model complexity and reduce the need for the model to learn from sequencing. 

Final model parameters:
Embedding dimension = 50
Linear layer dimension = 100 (with relu and dropout applied between layers)
Adam Optimizer with lr = 0.001
Epochs = 10 
Batch size = 20

The model could achieve 100% accuracy with under 1 minute of training.


#### RUN

````
python a2part2.py --train --text_path x_train.txt --label_path y_train.txt --model_path model.pt
````
````
python a2part2.py --test --text_path x_test.txt --model_path model.pt --output_path out.txt
````
````
python eval.py out.txt y_test.txt
````
