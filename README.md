# CAP4630-001_AI-LevelGen
AI models that generate 2D platformer levels from ASCII text using N-gram and LSTM architectures. Final project for CAP 4630 001

# AI for Generating Game Levels  
**CAP 4630 – Intro to Artificial Intelligence**  
**Team 25:** Tye Kates, Rodney Benjamin, Thomas Gregory, Jacob Eurglunes, Adam Kieszkowski

---

## 📌 Project Overview
This project explores how AI can generate new 2D platformer-style game levels using text-based ASCII tile maps.  
We compare two generative models:

- **3-gram Markov baseline**
- **Character-level LSTM neural network**

Both models are trained on a custom dataset of ASCII tile levels and evaluated using style similarity metrics and simple playability rules.

---

## 🎮 Dataset Description
The dataset contains handcrafted 2D platformer levels represented in plain text.  
Each tile uses one of the following characters:

| Symbol | Meaning |
|--------|----------|
| `X` | Ground tile |
| `-` | Empty space |
| `?` | Question block |
| `E` | Enemy |

All levels share a **fixed row width**, allowing them to form a clean rectangular layout.

Example snippet from the dataset:


---

## 🧠 Models Used

### **1️⃣ 3-gram Markov Model**
- Predicts each next character using the previous two characters.
- Very fast and simple.
- Captures short-term tile patterns.
- Struggles with global level structure.

### **2️⃣ LSTM (Long Short-Term Memory)**
- Character-level recurrent neural network.
- Learns longer-range structure and patterns.
- Produces more coherent and game-like layouts.
- Trained for 3 epochs on a 4,907-character corpus.

---

## 🔧 Project Structure

ai-levelgen/
│
├── data/
│ ├── raw/ # Original ASCII levels
│ └── processed/ # Combined dataset (dataset.txt)
│
├── models/
│ ├── ngram.pkl # Trained 3-gram model
│ └── lstm.pt # Trained LSTM weights
│
├── reports/
│ └── figures/ # Generated sample levels
│
├── src/
│ ├── prepare_data.py # Builds training corpus
│ ├── ngram.py # 3-gram training
│ ├── lstm_train.py # LSTM training script
│ ├── generate.py # Level generator
│ └── evaluate.py # KL + playability evaluation
│
└── Final Project Presentation.pptx


---

## ▶️ How to Run This Project (Google Colab or local Python)

### **1. Build the dataset**

python src/prepare_data.py \
--input_dir data/raw \
--out data/processed/dataset.txt

### **2. Train the 3-gram model**

python src/lstm_train.py \
--data data/processed/dataset.txt \
--epochs 3 \
--out models/lstm.pt

### **3. Train the LSTM**
python src/lstm_train.py \
--data data/processed/dataset.txt \
--epochs 3 \
--out models/lstm.pt

### **4. Generate levels (LSTM or N-gram)**

LSTM:

python src/generate.py \
--model lstm \
--ckpt models/lstm.pt \
--length 1200 \
--out reports/figures/sample_lstm.txt


N-gram:

python src/generate.py \
--model ngram \
--ckpt models/ngram.pkl \
--length 1200 \
--out reports/figures/sample_ngram.txt

### **5. Evaluate**
python src/evaluate.py \
--ref data/processed/dataset.txt \
--gen reports/figures/sample_lstm.txt \
--tile_set "X-?E" \
--width 60

📊 Results

### **Training Loss (LSTM)**
| Epoch | Loss   |
| ----- | ------ |
| 1     | 0.3532 |
| 2     | 0.1840 |
| 3     | 0.1700 |

KL Divergence (Style Similarity)

Result: 0.0513
A very low KL score indicates that the LSTM’s generated tile distribution is highly similar to the training data.

Playability Check

Playable: False

Reason: Inconsistent width across lines

This means the model generated correct style, but inconsistent structure.

📘 Sample Outputs

3-gram Output Snippet
(paste 4–8 lines from sample_ngram.txt here)

LSTM Output Snippet
(paste 4–8 lines from sample_lstm.txt here)

Interpretation

The 3-gram model captures small patterns but produces noisy, chaotic levels.

The LSTM generates coherent stretches of ground and patterns similar to the dataset, but needs structure constraints to guarantee playability.

⚠️ Challenges & Limitations

Very small dataset limits model performance.

3-gram model lacks global structure.

LSTM sometimes generates uneven line lengths.

No constraints or post-processing to enforce rectangular levels.

Limited time for tuning or experimenting with advanced models.

🏁 Conclusion & Future Work
Conclusion

LSTM outperforms the 3-gram baseline.

The model learns the visual style of the levels very well (low KL).

Structural constraints are needed for proper playability.

Future Work

Add post-processing to fix row width.

Train on a larger and more diverse dataset.

Explore transformer-based generative models.

Add constraint-based or reward-based playability checks.

### **Team**

👥 Team 25

Tye Kates

Rodney Benjamin

Thomas Gregory

Jacob Eurglunes

Adam Kieszkowski


📎 Additional Files

Final Project Presentation.pptx – Full slide deck

sample_lstm.txt / sample_ngram.txt – Generated levels

dataset.txt – Combined training corpus
