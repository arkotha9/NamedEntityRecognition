Python version used: Python 3.11.6

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