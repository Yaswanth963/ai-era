# Practical Coding Exercises & Projects
## Hands-On Learning for AI Engineering

This document contains **300+ coding exercises** organized by topic, progressing from beginner to advanced. Complete these alongside the main roadmap for maximum practical experience.

---

## MONTH 1: Python & Machine Learning Basics

### Week 1: NumPy & Pandas Exercises (50 exercises)

**NumPy (25 exercises)**:
1. Create arrays using different methods (arange, zeros, ones, linspace)
2. Array indexing and slicing (1D, 2D, 3D)
3. Boolean indexing and fancy indexing
4. Array reshaping and transposing
5. Broadcasting operations
6. Array concatenation and splitting
7. Statistical operations (mean, std, median)
8. Linear algebra operations (dot, matmul, inv)
9. Random number generation
10. Array sorting and searching
11. Create a multiplication table using broadcasting
12. Normalize an array (min-max scaling)
13. Find outliers using z-score
14. Implement moving average
15. Create correlation matrix
16. Generate synthetic dataset
17. Image manipulation with arrays
18. Implement convolution operation
19. Calculate covariance matrix
20. Implement PCA from scratch (basic version)
21. Create confusion matrix calculator
22. Implement cosine similarity
23. Calculate Euclidean distance matrix
24. Batch matrix operations
25. Memory-efficient array operations

**Pandas (25 exercises)**:
1. Load CSV, Excel, JSON files
2. DataFrame creation from dict, list, arrays
3. Data inspection (head, tail, info, describe)
4. Column selection and filtering
5. Row filtering with conditions
6. Handling missing values (dropna, fillna)
7. Data type conversions
8. Sorting and ranking
9. GroupBy operations
10. Aggregation functions
11. Merging and joining DataFrames
12. Pivot tables and cross-tabulation
13. String operations on columns
14. DateTime operations
15. Apply custom functions
16. Create derived columns
17. Handle duplicates
18. Categorical data encoding
19. Binning continuous variables
20. Rolling window operations
21. Data reshaping (melt, stack, unstack)
22. Multi-index DataFrames
23. Read and write various formats
24. Performance optimization techniques
25. Build ETL pipeline

**Mini Projects**:
- **Project 1**: Clean messy dataset (handle missing, outliers, types)
- **Project 2**: Exploratory Data Analysis (EDA) on Titanic dataset
- **Project 3**: Sales data analysis with visualizations

---

### Week 2-4: Machine Learning Exercises (75 exercises)

**Regression (15 exercises)**:
1. Simple linear regression from scratch
2. Multiple linear regression with scikit-learn
3. Polynomial regression
4. Ridge regression (L2 regularization)
5. Lasso regression (L1 regularization)
6. ElasticNet regression
7. Feature scaling (standardization, normalization)
8. Feature engineering for house prices
9. Residual analysis and diagnostics
10. Cross-validation for regression
11. Learning curves analysis
12. Hyperparameter tuning with GridSearch
13. Predict house prices (Kaggle dataset)
14. Time series forecasting with regression
15. Ensemble regression models

**Classification (20 exercises)**:
1. Logistic regression from scratch
2. Binary classification with scikit-learn
3. Multi-class classification
4. Decision tree classifier
5. Visualize decision boundaries
6. Random forest classifier
7. Feature importance analysis
8. Handle imbalanced datasets (SMOTE)
9. Confusion matrix and metrics
10. ROC curve and AUC calculation
11. Precision-recall curve
12. Cross-validation strategies
13. Stratified sampling
14. Calibration curves
15. Spam email classifier
16. Credit card fraud detection
17. Customer churn prediction
18. Iris species classification
19. Titanic survival prediction
20. Disease diagnosis prediction

**Unsupervised Learning (15 exercises)**:
1. K-Means clustering from scratch
2. K-Means with scikit-learn
3. Elbow method for optimal K
4. Hierarchical clustering
5. DBSCAN clustering
6. Gaussian Mixture Models
7. PCA from scratch
8. PCA with scikit-learn
9. t-SNE for visualization
10. Anomaly detection
11. Customer segmentation
12. Image compression with K-Means
13. Topic modeling basics
14. Dimensionality reduction comparison
15. Clustering evaluation metrics

**Model Evaluation & Selection (15 exercises)**:
1. Train-test split strategies
2. K-fold cross-validation
3. Stratified K-fold
4. Time series split
5. Hold-out validation
6. Calculate all classification metrics
7. Implement confusion matrix
8. ROC-AUC calculation
9. Learning curves
10. Validation curves
11. GridSearchCV for hyperparameters
12. RandomizedSearchCV
13. Nested cross-validation
14. Model comparison framework
15. Statistical significance testing

**Feature Engineering (10 exercises)**:
1. One-hot encoding
2. Label encoding
3. Ordinal encoding
4. Feature scaling techniques
5. Polynomial features
6. Interaction features
7. Date-time feature extraction
8. Text feature extraction (Bag of Words)
9. Target encoding
10. Feature selection methods

---

## MONTH 2: Deep Learning & Computer Vision

### Week 5: Neural Networks Exercises (40 exercises)

**PyTorch Basics (15 exercises)**:
1. Create tensors (various methods)
2. Tensor operations (add, multiply, matmul)
3. Tensor indexing and slicing
4. Reshape and view operations
5. GPU tensor operations
6. Autograd and gradients
7. Create custom autograd function
8. Build computational graph
9. Implement forward pass
10. Implement backward pass
11. Gradient descent from scratch
12. Mini-batch gradient descent
13. Optimizer comparison (SGD, Adam)
14. Learning rate scheduling
15. Save and load models

**Neural Networks (25 exercises)**:
1. Implement perceptron from scratch
2. Single-layer neural network
3. Multi-layer perceptron (MLP)
4. Activation functions (ReLU, Sigmoid, Tanh)
5. Implement forward propagation
6. Implement backpropagation
7. Weight initialization strategies
8. Batch normalization
9. Dropout regularization
10. Early stopping
11. Build custom nn.Module
12. Custom loss function
13. Custom optimizer
14. MNIST digit classification
15. Fashion-MNIST classification
16. Binary classification (NN)
17. Multi-class classification (NN)
18. Regression with neural networks
19. Overfitting vs underfitting experiments
20. Hyperparameter tuning (layers, neurons)
21. Learning rate finder
22. Gradient clipping
23. Model ensembling
24. Knowledge distillation
25. Neural network visualization

**Mini Projects**:
- **Project 1**: XOR problem with NN
- **Project 2**: MNIST 99%+ accuracy
- **Project 3**: Custom dataset classifier

---

### Week 6-8: Computer Vision Exercises (60 exercises)

**CNN Fundamentals (20 exercises)**:
1. Understand convolution operation
2. Implement 2D convolution from scratch
3. Pooling operations (max, average)
4. Build simple CNN
5. LeNet architecture
6. AlexNet architecture (simplified)
7. VGG architecture
8. ResNet skip connections
9. Inception modules
10. Batch normalization in CNNs
11. CIFAR-10 classification
12. CIFAR-100 classification
13. Data augmentation techniques
14. Visualize CNN filters
15. Visualize feature maps
16. Grad-CAM visualization
17. Transfer learning basics
18. Fine-tune pre-trained model
19. Feature extraction with CNN
20. CNN for regression tasks

**Advanced CV (20 exercises)**:
1. Object detection with YOLO
2. Train YOLO on custom dataset
3. Non-max suppression
4. Anchor boxes concept
5. Faster R-CNN implementation
6. Image segmentation with U-Net
7. Semantic segmentation
8. Instance segmentation
9. Mask R-CNN
10. Image classification API
11. Face detection
12. Face recognition system
13. Facial landmark detection
14. Pose estimation
15. Image captioning
16. Style transfer
17. Super-resolution
18. Image denoising
19. OCR implementation
20. Real-time object tracking

**Practical CV Projects (20 exercises)**:
1. Build custom dataset loader
2. Image augmentation pipeline
3. Multi-label classification
4. Fine-grained classification
5. Zero-shot learning
6. Few-shot learning
7. Siamese networks
8. Triplet loss implementation
9. Metric learning
10. Attention mechanisms in vision
11. Vision Transformers (ViT)
12. CLIP model usage
13. Object detection API
14. Webcam real-time detection
15. Video classification
16. Action recognition
17. Image quality assessment
18. Anomaly detection in images
19. Medical image analysis
20. Satellite image classification

---

## MONTH 3: NLP & LLMs

### Week 9-10: NLP Fundamentals (50 exercises)

**Text Processing (15 exercises)**:
1. Tokenization (word, sentence)
2. Stemming and lemmatization
3. Stop words removal
4. Part-of-speech tagging
5. Named entity recognition
6. Dependency parsing
7. Text normalization
8. Regular expressions for text
9. Unicode handling
10. Text cleaning pipeline
11. Language detection
12. Spell checking
13. Text statistics
14. Readability scores
15. Text preprocessing pipeline

**Word Embeddings (15 exercises)**:
1. One-hot encoding
2. Bag of Words (BoW)
3. TF-IDF from scratch
4. TF-IDF with sklearn
5. Word2Vec training
6. Word2Vec similarity
7. Word analogies
8. GloVe embeddings
9. FastText embeddings
10. Contextualized embeddings
11. Embedding visualization (t-SNE)
12. Document embeddings
13. Sentence embeddings
14. Embedding arithmetic
15. Embedding evaluation

**Sequence Models (20 exercises)**:
1. RNN from scratch
2. RNN with PyTorch
3. LSTM implementation
4. GRU implementation
5. Bidirectional RNN
6. Sequence classification
7. Sentiment analysis
8. Text generation with RNN
9. Language modeling
10. Sequence-to-sequence model
11. Attention mechanism
12. Text summarization
13. Machine translation
14. Question answering
15. Intent classification
16. Named entity recognition
17. Text classification pipeline
18. Multi-label text classification
19. Hierarchical classification
20. Aspect-based sentiment analysis

---

### Week 11-12: Transformers & LLMs (50 exercises)

**Transformer Fundamentals (20 exercises)**:
1. Self-attention mechanism
2. Multi-head attention
3. Positional encoding
4. Transformer encoder
5. Transformer decoder
6. BERT architecture
7. GPT architecture
8. T5 architecture
9. Tokenization (BPE, WordPiece)
10. Attention visualization
11. Load pre-trained BERT
12. Load pre-trained GPT
13. Extract embeddings
14. Fine-tune for classification
15. Fine-tune for NER
16. Fine-tune for QA
17. Zero-shot classification
18. Few-shot learning
19. Prompt engineering
20. Chain-of-thought prompting

**Hugging Face Ecosystem (15 exercises)**:
1. Transformers library basics
2. AutoModel and AutoTokenizer
3. Pipeline API for tasks
4. Custom dataset loading
5. Trainer API
6. Training arguments configuration
7. Evaluation metrics
8. Model checkpointing
9. Push to Hub
10. Load from Hub
11. Model quantization
12. ONNX export
13. TensorRT optimization
14. Multi-GPU training
15. Distributed training

**LLM Applications (15 exercises)**:
1. Text classification with BERT
2. Sentiment analysis API
3. Named entity recognition
4. Question answering system
5. Text summarization
6. Text generation with GPT
7. Chatbot with GPT
8. Code generation
9. SQL query generation
10. Prompt optimization
11. Few-shot prompting
12. Chain prompting
13. LLM evaluation
14. Toxicity detection
15. Fact-checking system

---

## MONTH 4: RAG & MLOps

### Week 13-14: Vector DBs & RAG (40 exercises)

**Embeddings & Search (15 exercises)**:
1. Sentence transformers
2. Generate embeddings
3. Cosine similarity search
4. FAISS index creation
5. FAISS similarity search
6. ChromaDB setup
7. Store embeddings in ChromaDB
8. Query vector database
9. Hybrid search (keyword + semantic)
10. Re-ranking results
11. Embedding model comparison
12. Dense retrieval
13. Sparse retrieval (BM25)
14. Multi-vector retrieval
15. Embedding visualization

**RAG Implementation (25 exercises)**:
1. Document loader (PDF)
2. Document loader (Word, HTML)
3. Text splitting strategies
4. Recursive character splitter
5. Semantic chunking
6. Metadata extraction
7. Build basic RAG
8. RAG with LangChain
9. RAG with LlamaIndex
10. Conversational RAG
11. Multi-document RAG
12. RAG with sources
13. RAG with citations
14. Re-ranking in RAG
15. Query rewriting
16. Hypothetical document embeddings
17. RAG evaluation metrics
18. A/B test RAG systems
19. Multi-modal RAG
20. RAG with images
21. RAG with tables
22. RAG optimization
23. Caching strategies
24. Production RAG pipeline
25. RAG monitoring

---

### Week 15-16: MLOps (40 exercises)

**Model Serving (15 exercises)**:
1. Save model (pickle, joblib)
2. Load and predict
3. Flask API basics
4. FastAPI basics
5. Serve scikit-learn model
6. Serve PyTorch model
7. Serve Hugging Face model
8. Request/response validation
9. Error handling
10. API documentation (Swagger)
11. Authentication
12. Rate limiting
13. Logging
14. Load testing (Locust)
15. Async API endpoints

**Containerization (10 exercises)**:
1. Write Dockerfile for ML app
2. Build Docker image
3. Run container locally
4. Multi-stage builds
5. Docker Compose
6. Environment variables
7. Volume mounting
8. Docker networking
9. Push to DockerHub
10. Docker best practices

**MLOps Tools (15 exercises)**:
1. MLflow tracking
2. Log parameters and metrics
3. Log artifacts
4. Model registry
5. MLflow projects
6. DVC setup
7. Track data with DVC
8. Version datasets
9. GitHub Actions for ML
10. Automated testing
11. Model monitoring
12. Data drift detection
13. Model drift detection
14. Alerting system
15. A/B testing framework

---

## MONTH 5: Agents & MCP

### Week 17-18: Agentic AI (50 exercises)

**Agent Basics (20 exercises)**:
1. Simple tool-using agent
2. ReAct pattern implementation
3. LangChain agent with tools
4. Custom tool creation
5. Wikipedia search tool
6. Calculator tool
7. Web scraping tool
8. File system tool
9. SQL database tool
10. API calling tool
11. Zero-shot agent
12. Conversational agent
13. Agent with memory
14. Multi-step reasoning
15. Agent planning
16. Self-ask agent
17. Plan-and-execute agent
18. Agent debugging
19. Agent evaluation
20. Agent optimization

**Advanced Agents (20 exercises)**:
1. Research agent
2. Code generation agent
3. Debugging agent
4. Data analysis agent
5. Writing assistant agent
6. Customer service agent
7. Personal assistant agent
8. Multi-agent system
9. Agent orchestration
10. Collaborative agents
11. Competitive agents
12. Hierarchical agents
13. Agent with function calling
14. OpenAI Assistants API
15. Anthropic Claude tools
16. Agent with code interpreter
17. Agent with file search
18. Agent persistence
19. Human-in-the-loop agent
20. Production agent system

**Agent Projects (10 exercises)**:
1. Autonomous web researcher
2. Code reviewer agent
3. Content generator agent
4. Email automation agent
5. Task automation agent
6. Meeting scheduler agent
7. Report generator agent
8. Data pipeline agent
9. Testing agent
10. DevOps agent

---

### Week 19-20: MCP & Multi-modal (40 exercises)

**Model Context Protocol (20 exercises)**:
1. MCP specification study
2. Simple MCP server
3. Simple MCP client
4. Tool registration in MCP
5. Resource management
6. Context passing
7. Multi-tool MCP server
8. MCP with database
9. MCP with file system
10. MCP with APIs
11. MCP error handling
12. MCP authentication
13. MCP rate limiting
14. MCP logging
15. MCP testing
16. Multiple MCP servers
17. MCP client orchestration
18. MCP monitoring
19. MCP documentation
20. Production MCP system

**Multi-modal AI (20 exercises)**:
1. CLIP model usage
2. Image-text matching
3. Zero-shot image classification
4. Image captioning
5. Visual question answering
6. LLaVA model usage
7. Image understanding with LLM
8. Text-to-image (Stable Diffusion)
9. Image editing with diffusion
10. Whisper speech-to-text
11. Text-to-speech
12. Voice cloning
13. Audio classification
14. Music generation
15. Video understanding
16. Video captioning
17. Multi-modal embeddings
18. Multi-modal search
19. Multi-modal RAG
20. Complete voice assistant

---

## MONTH 6: Portfolio Projects

### Week 21-22: Capstone Projects

**Project A: Enterprise RAG System**
- [ ] Document ingestion (PDF, Word, HTML)
- [ ] Advanced chunking (semantic, recursive)
- [ ] Vector database (Pinecone/ChromaDB)
- [ ] Re-ranking system
- [ ] Conversational interface
- [ ] Citation tracking
- [ ] Multi-language support
- [ ] Analytics dashboard
- [ ] User authentication
- [ ] Deployment to cloud

**Project B: Autonomous Agent Platform**
- [ ] Multi-agent framework
- [ ] Agent orchestration
- [ ] MCP integration
- [ ] Tool ecosystem (10+ tools)
- [ ] Planning system
- [ ] Memory management
- [ ] Web interface
- [ ] API for agents
- [ ] Monitoring dashboard
- [ ] Demo video

**Project C: Production ML Pipeline**
- [ ] Data ingestion
- [ ] Feature engineering
- [ ] Model training
- [ ] Hyperparameter tuning
- [ ] Model evaluation
- [ ] A/B testing
- [ ] Deployment pipeline
- [ ] Monitoring system
- [ ] Auto-retraining
- [ ] Alerting

---

## LeetCode Practice Schedule

### Week 23: Coding Interview Prep (100 problems)

**Arrays (25 problems)**:
- Two Sum, Three Sum
- Best Time to Buy/Sell Stock
- Container With Most Water
- Merge Intervals
- Rotate Array
- Product of Array Except Self
- Maximum Subarray
- Find Minimum in Rotated Array
- Search in Rotated Array
- Spiral Matrix
- Set Matrix Zeroes
- Game of Life
- Plus One
- Pascal's Triangle
- Remove Duplicates
- Move Zeroes
- Contains Duplicate
- Single Number
- Intersection of Arrays
- Missing Number
- Find All Duplicates
- Majority Element
- Sort Colors
- Next Permutation
- Jump Game

**Strings (25 problems)**:
- Valid Palindrome
- Longest Substring Without Repeating
- Longest Palindromic Substring
- Valid Anagram
- Group Anagrams
- String to Integer (atoi)
- Implement strStr()
- Count and Say
- Longest Common Prefix
- Valid Parentheses
- Generate Parentheses
- Letter Combinations of Phone Number
- Decode Ways
- Word Break
- Word Search
- Palindrome Partitioning
- Edit Distance
- Regular Expression Matching
- Wildcard Matching
- Minimum Window Substring
- Substring with Concatenation
- Text Justification
- Basic Calculator
- Integer to Roman
- Roman to Integer

**Hash Tables (15 problems)**:
- Two Sum
- Group Anagrams
- Top K Frequent Elements
- Valid Sudoku
- Happy Number
- Isomorphic Strings
- Word Pattern
- Contains Duplicate II
- Longest Consecutive Sequence
- Minimum Window Substring
- Fraction to Recurring Decimal
- Max Points on a Line
- Insert Delete GetRandom O(1)
- LRU Cache
- All O'one Data Structure

**Trees (20 problems)**:
- Maximum Depth of Binary Tree
- Validate Binary Search Tree
- Symmetric Tree
- Binary Tree Level Order Traversal
- Invert Binary Tree
- Lowest Common Ancestor
- Serialize and Deserialize Binary Tree
- Construct Binary Tree from Preorder/Inorder
- Flatten Binary Tree to Linked List
- Path Sum
- Binary Tree Maximum Path Sum
- Count Complete Tree Nodes
- Kth Smallest Element in BST
- Implement Trie
- Word Search II
- Binary Tree Right Side View
- Average of Levels
- Binary Tree Zigzag Traversal
- Recover Binary Search Tree
- Populating Next Right Pointers

**Dynamic Programming (15 problems)**:
- Climbing Stairs
- House Robber
- Coin Change
- Longest Increasing Subsequence
- Longest Common Subsequence
- Edit Distance
- Decode Ways
- Unique Paths
- Minimum Path Sum
- Jump Game
- Word Break
- Partition Equal Subset Sum
- Maximum Product Subarray
- Best Time to Buy/Sell Stock with Cooldown
- Burst Balloons

---

## Progress Tracking

### Completion Checklist

**Month 1 Exercises**:
- [ ] NumPy: 25/25
- [ ] Pandas: 25/25
- [ ] Regression: 15/15
- [ ] Classification: 20/20
- [ ] Unsupervised: 15/15

**Month 2 Exercises**:
- [ ] PyTorch: 15/15
- [ ] Neural Networks: 25/25
- [ ] CNNs: 20/20
- [ ] Advanced CV: 20/20

**Month 3 Exercises**:
- [ ] Text Processing: 15/15
- [ ] Word Embeddings: 15/15
- [ ] Sequence Models: 20/20
- [ ] Transformers: 20/20
- [ ] Hugging Face: 15/15

**Month 4 Exercises**:
- [ ] Embeddings: 15/15
- [ ] RAG: 25/25
- [ ] Model Serving: 15/15
- [ ] MLOps: 25/25

**Month 5 Exercises**:
- [ ] Agents: 40/40
- [ ] MCP: 20/20
- [ ] Multi-modal: 20/20

**Month 6**:
- [ ] Capstone projects: 3/3
- [ ] LeetCode: 100/100

**Total**: _____ / 650+ exercises

---

## Resources for Practice

### Platforms:
- **Kaggle**: Datasets and competitions
- **LeetCode**: Coding practice
- **HackerRank**: ML challenges
- **GitHub**: Study real projects
- **Hugging Face**: Models and datasets

### Daily Practice Routine:
- **Morning (2 hours)**: Theory + small exercises
- **Afternoon (3 hours)**: Major exercises + debugging
- **Evening (2 hours)**: Project work

---

**Remember**: The key is to CODE EVERY DAY. Theory without practice is useless. Build, break, debug, and ship! 🚀
