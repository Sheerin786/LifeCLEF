# LifeCLEF
**SHEERINHARINETCE at LifeCLEF: Deep Visual Embeddings and FAISS based Accelerated Similarity Search for Species Identification**
This project represents retrieval-based species identification approach developed for the AnimalCLEF 2025 challenge, leveraging deep visual embeddings extracted using EfficientNet-B4 and Vision Transformers (ViT). These embeddings are indexed using FAISS for efficient similarity search. In this, feature vectors are indexed using Facebook AI Similarity Search (FAISS) structures such as IndexFlatL2 and IndexIVFFlat, enabling scalable and efficient similarity searches during inference. In the training phase, image preprocessing, data augmentation, and model fine-tuning are performed to generate discriminative features, which are then indexed along with their labels. During testing, features of query images are matched against the indexed dataset, and species labels are predicted through k-nearest neighbor search followed by majority voting. Labels with confidence scores below a predefined threshold are rejected to enhance prediction reliability. Experimental results across multiple runs show that Vision Transformer excels in capturing fine-grained, global image relationships, while EfficientNet performs efficiently with fewer parameters. The best configuration achieved a top-5 accuracy of 0.34 and a top-1 accuracy (BAKS) of 0.27 on the private leaderboard. We highlight the trade-offs of approximate nearest-neighbor search in smaller datasets, analyze model behavior through limited qualitative examples, and discuss challenges such as class imbalance, occlusion, and fine-grained distinctions. Limitations include relatively modest scores and reproducibility concerns. 


Description:
Pre-process.ipynb - Preprocess the dataset
Program.py - Main core (FAISS with EfficientNet)
Postprocessing.ipynb - The output generated from program.py is pstrocessed here.

Detailed description about the work will be available in the manuscript.
