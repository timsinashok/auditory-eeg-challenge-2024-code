# Auditory EEG Challenge 2024

This repository houses my contributions to the Auditory EEG Challenge 2024, specifically focusing on Task 1, named **Match Mismatch**.

### 1. Goal and my work Task 1 - Match Mismatch
- **Task Description:** In Task 1, the goal was to address the challenges related to matching and mismatching auditory stimuli based on EEG data. The challenge was to train a machine-learning model that would identify which auditory signal created the given EEG Signal.  

- **My Work:**
  - Implemented and experimented with various machine learning models.
  - Fine-tuned the models for optimal performance on the given dataset.

- **Results:**
  - Achieved validation accuracy of 63% on the validation dataset using the combination of dilated CNN and LSTM with normalization


### 2. Instructions to Run the Program

To run the program and reproduce the results, follow these steps:

1. **Download the Data:**
   - Obtain the dataset required for the challenge.
   - Place the downloaded data into the `split_data` folder in the root directory.

2. **Choose a Model:**
   - Navigate to the `experiments` folder to explore available models.

3. **Run the Program:**
   - Select a specific model script to execute.
   - Run the chosen script using a command-line interface or an integrated development environment.

   Example (using the command line):
   ```bash
   python experiments/model_name.py
