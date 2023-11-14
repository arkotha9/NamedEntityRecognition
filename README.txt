Python version used: Python 3.11.6

- I performed Named Entity Recognition on general and glove word embeddings. As embeddings as generally case insensitive, I added three additional features for each token embedding(isTitle, isUpper,isLower) to make my BiLSTM model and NER case-sensitive. 
- I achieved f1 scores of around 80% for my generic 'conll2023' word embeddings and f1 scores of around 90% for my GloVe embeddings. 
-The superior performance of the BiLSTM with GloVe embeddings can be attributed to the embeddings' ability to capture intricate semantic relationships and contextual information derived from extensive and diverse textual datasets. GloVe's pre-training ensures a robust understanding of language semantics, enabling the model to discern nuanced meanings and context. Furthermore, the embeddings offer domain-general knowledge, facilitating adaptability across different contexts and tasks. Their reduced dimensionality contributes to faster training and improved generalization, particularly beneficial when dealing with limited data. Additionally, GloVe embeddings inherently encapsulate syntactic and grammatical structures, enhancing the model's comprehension of language syntax. In essence, leveraging GloVe embeddings enriches the BiLSTM's understanding of input data, leading to superior performance compared to models without such pre-trained embeddings.


The zip file is organized as:
.
├── HW4_Ananya_Kotha.ipynb
├── HW4_Ananya_Kotha.pdf
├── bilstm1_state_dict.pt (stored the first model as .pt file)
├── bilstm2_state_dict.pt (stored the second model as .pt file)
├── conlleval.py
├── eval_task1.py
├── eval_task2.py
└── README.txt

Assuming we are in current working directory and it has the above file structure with glove embeddings in the same folder,
1) pip install datasets
2) pip install torch

To produce results on test data for task 1:
-> $python eval_task1.py

To produce results on test data for task 2:
—> $python eval_task2.py
