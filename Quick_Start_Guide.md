# Quick Start Guide
## Get Started Today - AI Engineer 2026 Roadmap

This guide will help you set up your development environment and start coding within 1-2 hours.

---

## 🚀 Day 1: Setup & First Steps

### Step 1: Install Python (15 minutes)

**macOS** (you're on Mac):
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11 (recommended for AI/ML)
brew install python@3.11

# Verify installation
python3.11 --version
```

**Alternative**: Download from [python.org](https://www.python.org/downloads/)

---

### Step 2: Set Up Virtual Environment (10 minutes)

```bash
# Create project directory
mkdir ai-learning-2026
cd ai-learning-2026

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
```

---

### Step 3: Install Core Libraries (20 minutes)

```bash
# Upgrade pip
pip install --upgrade pip

# Week 1 essentials
pip install numpy pandas matplotlib seaborn jupyter scikit-learn

# Verify installations
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import sklearn; print('Scikit-learn:', sklearn.__version__)"
```

---

### Step 4: Set Up Development Tools (30 minutes)

**A. Install VS Code**:
- Download from [code.visualstudio.com](https://code.visualstudio.com/)
- Install Python extension (Microsoft)
- Install Jupyter extension (Microsoft)

**B. Configure VS Code**:
```json
// File -> Preferences -> Settings (JSON)
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "jupyter.jupyterServerType": "local",
    "python.linting.enabled": true,
    "python.formatting.provider": "black"
}
```

**C. Install Git**:
```bash
# macOS (if not installed)
brew install git

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

### Step 5: Create Accounts (20 minutes)

1. **GitHub**: [github.com](https://github.com)
   - Create account
   - Set up profile
   - Enable 2FA

2. **Kaggle**: [kaggle.com](https://kaggle.com)
   - Create account
   - Verify email
   - Join competitions

3. **Hugging Face**: [huggingface.co](https://huggingface.co)
   - Create account
   - Get API token (for later)

4. **OpenAI** (optional for now): [platform.openai.com](https://platform.openai.com)
   - Create account
   - Add $5 credit (when you need it)

---

### Step 6: First Jupyter Notebook (15 minutes)

```bash
# Create notebooks directory
mkdir notebooks
cd notebooks

# Start Jupyter
jupyter notebook
```

**Create your first notebook** (`Day1_Setup.ipynb`):

```python
# Cell 1: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

print("✅ All imports successful!")

# Cell 2: NumPy basics
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {arr.mean()}")
print(f"Sum: {arr.sum()}")

# Cell 3: Pandas basics
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [85, 90, 95]
})
print(df)

# Cell 4: Visualization
iris = load_iris()
plt.figure(figsize=(10, 6))
sns.scatterplot(x=iris.data[:, 0], y=iris.data[:, 1], 
                hue=iris.target, palette='viridis')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset')
plt.show()

# Cell 5: First ML Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎉 Your first ML model accuracy: {accuracy:.2%}")
```

---

## 📚 Essential Resources (Bookmark These)

### Learning Platforms

**Free & Best**:
1. **Fast.ai** - [course.fast.ai](https://course.fast.ai)
   - Practical Deep Learning for Coders
   - Start Week 2 of Month 2

2. **Kaggle Learn** - [kaggle.com/learn](https://www.kaggle.com/learn)
   - Python, ML, DL courses
   - Start immediately

3. **Hugging Face Course** - [huggingface.co/course](https://huggingface.co/course)
   - NLP with Transformers
   - Start Month 3

4. **DeepLearning.AI** - [deeplearning.ai](https://www.deeplearning.ai)
   - ML Specialization (Coursera)
   - DL Specialization (Coursera)
   - LangChain courses (free on site)

5. **Made With ML** - [madewithml.com](https://madewithml.com)
   - MLOps course
   - Start Month 4

**YouTube Channels**:
- **3Blue1Brown**: Neural networks explained visually
- **Andrej Karpathy**: Deep learning fundamentals
- **StatQuest**: ML concepts simplified
- **Sentdex**: Python ML tutorials
- **Two Minute Papers**: Latest AI research

---

### Documentation (Always Open)

1. **NumPy**: [numpy.org/doc](https://numpy.org/doc/stable/)
2. **Pandas**: [pandas.pydata.org](https://pandas.pydata.org/docs/)
3. **Scikit-learn**: [scikit-learn.org](https://scikit-learn.org/stable/)
4. **PyTorch**: [pytorch.org/docs](https://pytorch.org/docs/stable/index.html)
5. **Hugging Face**: [huggingface.co/docs](https://huggingface.co/docs)
6. **LangChain**: [python.langchain.com](https://python.langchain.com/docs/get_started/introduction)

---

### Books to Download

**Must-Read** (in order):
1. **Hands-On Machine Learning** (Aurélien Géron)
   - Month 1-2
   - [GitHub repo](https://github.com/ageron/handson-ml3)

2. **Deep Learning with PyTorch** (Manning)
   - Month 2
   - [Free chapters](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)

3. **Natural Language Processing with Transformers** (O'Reilly)
   - Month 3
   - [GitHub repo](https://github.com/nlp-with-transformers/notebooks)

4. **Designing Machine Learning Systems** (Chip Huyen)
   - Month 4
   - Best MLOps book

5. **Building LLM Apps** - Online resources
   - Month 4-5
   - Various blog posts and tutorials

---

## 📊 Week 1 Practice Plan

### Day 1 (Today): Setup + NumPy Basics
**Time**: 6-7 hours

**Morning (2-3 hours)**:
- [ ] Complete setup (above steps)
- [ ] NumPy tutorial: [Official Quickstart](https://numpy.org/doc/stable/user/quickstart.html)
- [ ] Practice: Create arrays, indexing, slicing

**Afternoon (2-3 hours)**:
- [ ] NumPy operations: broadcasting, reshaping
- [ ] Practice: 10 NumPy exercises from coding syllabus
- [ ] Watch: [NumPy crash course](https://www.youtube.com/watch?v=QUT1VHiLmmI)

**Evening (1-2 hours)**:
- [ ] Jupyter notebook with NumPy examples
- [ ] Push to GitHub (create repository)
- [ ] Document learnings

**Deliverable**: GitHub repo with NumPy practice notebook

---

### Day 2: NumPy Advanced + Pandas Intro
**Time**: 6-7 hours

**Morning**:
- [ ] NumPy linear algebra operations
- [ ] Statistical functions
- [ ] Practice: 10 more NumPy exercises

**Afternoon**:
- [ ] Pandas tutorial: [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [ ] DataFrames and Series
- [ ] Reading CSV files

**Evening**:
- [ ] Download Titanic dataset from Kaggle
- [ ] Load and explore with Pandas
- [ ] Create visualization

**Deliverable**: Titanic EDA notebook

---

### Day 3: Pandas Deep Dive
**Time**: 6-7 hours

**Morning**:
- [ ] Data cleaning techniques
- [ ] Handling missing values
- [ ] Data type conversions

**Afternoon**:
- [ ] GroupBy operations
- [ ] Aggregations
- [ ] Merging DataFrames

**Evening**:
- [ ] Practice: Complete data cleaning pipeline
- [ ] Kaggle dataset: [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- [ ] Initial EDA

**Deliverable**: Data cleaning notebook

---

### Day 4: Visualization
**Time**: 6-7 hours

**Morning**:
- [ ] Matplotlib basics
- [ ] Line plots, scatter plots, histograms
- [ ] Customization

**Afternoon**:
- [ ] Seaborn for statistical plots
- [ ] Heatmaps, pair plots
- [ ] Style and themes

**Evening**:
- [ ] Create 10 different visualizations
- [ ] Visualization portfolio
- [ ] Blog post (optional)

**Deliverable**: Visualization portfolio notebook

---

### Day 5: First ML Model
**Time**: 6-7 hours

**Morning**:
- [ ] ML concepts: supervised vs unsupervised
- [ ] Train-test split
- [ ] Linear regression theory

**Afternoon**:
- [ ] Implement linear regression (scikit-learn)
- [ ] Predict house prices
- [ ] Evaluate model

**Evening**:
- [ ] Feature engineering
- [ ] Improve model
- [ ] Document results

**Deliverable**: First ML project

---

### Day 6: Classification
**Time**: 6-7 hours

**Morning**:
- [ ] Logistic regression theory
- [ ] Binary classification
- [ ] Model evaluation metrics

**Afternoon**:
- [ ] Titanic survival prediction
- [ ] Confusion matrix
- [ ] ROC-AUC curve

**Evening**:
- [ ] Submit to Kaggle competition
- [ ] Refine and improve
- [ ] Week 1 review

**Deliverable**: Kaggle submission

---

## 🗓️ Month 1 Overview

### Weekly Goals:
- **Week 1**: Python, NumPy, Pandas, Viz → 3 projects
- **Week 2**: ML basics, Git, Jupyter → 3 projects
- **Week 3**: Classical ML algorithms → 3 projects
- **Week 4**: Unsupervised learning, deployment → 2 projects + 1 app

### Expected Output:
- 10-12 GitHub repositories
- 1-2 deployed web apps (Streamlit)
- 1-2 Kaggle competition entries
- 1-2 blog posts

---

## 💻 Development Environment Setup

### Create Project Structure:

```bash
ai-learning-2026/
├── venv/                      # Virtual environment
├── notebooks/                 # Jupyter notebooks
│   ├── week1/
│   ├── week2/
│   └── ...
├── projects/                  # Standalone projects
│   ├── project1-house-prices/
│   ├── project2-sentiment/
│   └── ...
├── exercises/                 # Practice exercises
│   ├── numpy_exercises.py
│   ├── pandas_exercises.py
│   └── ...
├── datasets/                  # Local datasets
├── models/                    # Saved models
├── scripts/                   # Utility scripts
├── docs/                      # Documentation
├── requirements.txt           # Dependencies
└── README.md                  # Project overview
```

### Create README Template:

```markdown
# AI Engineer Learning Journey 2026

## About
6-month roadmap to Senior AI Engineer role in top MNCs.

## Progress
- **Month 1**: [█████░░░░░] 50% - Python & ML Basics
- **Projects Completed**: 6/12
- **Skills**: NumPy, Pandas, Scikit-learn

## Projects
1. [House Price Predictor](./projects/project1-house-prices/)
2. [Sentiment Analyzer](./projects/project2-sentiment/)
...

## Blog Posts
1. [My First ML Model](link)
2. [Feature Engineering Tips](link)

## Connect
- LinkedIn: [Your Profile]
- Kaggle: [Your Profile]
- Email: your.email@example.com
```

---

## 🎯 Success Metrics

### Daily Tracking:
- [ ] 6-7 hours coding time
- [ ] 1-2 exercises completed
- [ ] Git commit(s) made
- [ ] Notes/documentation updated

### Weekly Tracking:
- [ ] 2-3 projects completed
- [ ] 1 GitHub repo created
- [ ] 10-15 exercises done
- [ ] Weekly reflection written

### Monthly Tracking:
- [ ] 8-12 projects completed
- [ ] 4-6 GitHub repos
- [ ] 1-2 blog posts
- [ ] Skill level assessment

---

## 🛠️ Troubleshooting

### Common Issues:

**1. Import errors**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall package
pip install --upgrade <package-name>
```

**2. Jupyter kernel issues**:
```bash
# Install ipykernel
pip install ipykernel

# Add virtual environment to Jupyter
python -m ipykernel install --user --name=venv
```

**3. Git issues**:
```bash
# Set up SSH keys (recommended)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Add to GitHub: Settings -> SSH Keys
```

**4. Memory errors (large datasets)**:
```python
# Read in chunks
df = pd.read_csv('large_file.csv', chunksize=10000)

# Use less memory
df = pd.read_csv('file.csv', dtype={'col': 'int32'})
```

---

## 📱 Recommended VS Code Extensions

```bash
# Python
code --install-extension ms-python.python

# Jupyter
code --install-extension ms-toolsai.jupyter

# Git
code --install-extension eamodio.gitlens

# Markdown
code --install-extension yzhang.markdown-all-in-one

# Code formatting
code --install-extension ms-python.black-formatter

# AI assistance (optional)
code --install-extension GitHub.copilot
```

---

## 🌐 Community & Support

### Join These Communities:

1. **Reddit**:
   - r/learnmachinelearning
   - r/MachineLearning
   - r/datascience
   - r/learnpython

2. **Discord**:
   - Hugging Face Discord
   - Fast.ai Discord
   - MLOps Community

3. **Slack**:
   - MLOps Community Slack
   - PyTorch Slack

4. **LinkedIn**:
   - Follow AI/ML influencers
   - Join AI/ML groups
   - Post your projects

### Getting Help:

1. **Stack Overflow**: For coding issues
2. **GitHub Issues**: For library-specific problems
3. **Discord/Slack**: For community support
4. **Study Groups**: Find or create one

---

## 📧 Weekly Accountability

### Template for Weekly Update (post on LinkedIn):

```
Week X Update - AI Engineer Journey 🚀

This week I:
✅ Completed [X] projects: [links]
✅ Learned: [key concepts]
✅ Deployed: [app link]
✅ Challenges: [what was hard]

Next week:
🎯 [Goal 1]
🎯 [Goal 2]

#MachineLearning #AI #100DaysOfCode #LearnInPublic
```

---

## 🎓 First Week Checklist

**Before you start Week 2:**

- [ ] Development environment set up
- [ ] All accounts created (GitHub, Kaggle, HF)
- [ ] NumPy proficiency: 7/10
- [ ] Pandas proficiency: 7/10
- [ ] Matplotlib/Seaborn: 6/10
- [ ] First ML model trained
- [ ] 3 notebooks on GitHub
- [ ] Titanic dataset explored
- [ ] House prices EDA completed
- [ ] First Kaggle submission
- [ ] VS Code configured
- [ ] Git workflow comfortable
- [ ] Learning routine established

---

## 💡 Pro Tips for Success

### 1. **Code Every Day**
- Minimum 30 minutes, even on busy days
- Consistency beats intensity

### 2. **Build in Public**
- Share progress on LinkedIn/Twitter
- Document learnings in blog posts
- Get feedback from community

### 3. **Focus on Projects, Not Tutorials**
- 70% coding, 30% watching/reading
- Build while you learn

### 4. **Don't Get Stuck in Tutorial Hell**
- Move on if spending >30 min on one concept
- Come back later with fresh perspective

### 5. **Version Everything**
- Commit daily to GitHub
- Use meaningful commit messages
- Organize repositories well

### 6. **Ask for Help**
- Don't struggle alone for hours
- Community is friendly and helpful
- Document solutions for others

### 7. **Review and Reflect**
- End-of-day: What did I learn?
- End-of-week: What went well? What didn't?
- Adjust plan as needed

### 8. **Rest and Recharge**
- Take one day off per week
- Avoid burnout
- Marathon, not sprint

---

## 🚀 You're Ready!

Everything you need is now set up. Here's what to do RIGHT NOW:

1. **Create your first notebook** (see Step 6 above)
2. **Run all cells** and see your first ML model work
3. **Push to GitHub** (create repository: "ai-learning-2026")
4. **Post on LinkedIn**: "Starting my AI Engineer journey! Day 1 ✅ #MachineLearning"

**Remember**: The best time to start was yesterday. The second best time is NOW.

Every expert was once a beginner. You've got this! 💪

---

## 📞 Need Help?

If you get stuck:
1. Check documentation
2. Search Stack Overflow
3. Ask in Reddit/Discord
4. Google the error message
5. Debug step by step

**You're not alone on this journey!**

---

**Start coding NOW! See you at the finish line in 6 months! 🎯**

*Good luck! 🚀*
