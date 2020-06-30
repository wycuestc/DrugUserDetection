Detection of Opioid Addicts via Attention-based Bidirectional Recurrent Neural Network
Quick Start
	1. Ensure your Python version is above 3.0, Tensorflow (2.2.0), Keras (2.3.1).
	2. Creating a file folder named "embedding_glove".
	3. Download “glove.6B.zip” from website https://nlp.stanford.edu/projects/glove/. Extract this file in the folder 		  “embedding_glove”.
	4. The main file is "biWordEmbeddingModify.py". You can modify hyperparameters, including max_length, embeddingDim, variantName, in the main function. Specifically, embeddingDim can be chosen as 50, 100, 200, 300. VariantName can be from "variant1" to "variant6", which is corresponding to the six model variants in my paper. max_length was chosen as 100 in my paper.
	5. The comparing methods are implemented in the file "comparisonBi.py".
