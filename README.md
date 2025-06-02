This project presents a machine learning-based approach for the forensic classification and detection of the NotPetya ransomware. By combining static and dynamic malware analysis techniques with feature engineering (API calls, opcode sequences, system artifacts), we build an ML model capable of distinguishing NotPetya from Petya and benign files.

Project Overview
- Reverse engineered ransomware samples using tools like Ghidra, Binwalk and Regshot.
- Extracted features: API calls, opcode sequences, disk-level behaviors, and obfuscation traits.
- Built and labeled a dataset based on extracted behaviors.
- Trained a Random Forest model with >90% accuracy in classifying NotPetya samples.

Directory Structure
- report: Final project report PDF.
- dataset: CSV dataset of extracted features.
- scripts: Python scripts for analysis and model training.
- models: Trained model artifacts.
- requirements.txt: Python dependencies.

Technologies Used
- Python 3
- Scikit-learn
- Ghidra
- REMnux
- Wireshark/TShark
- Regshot




