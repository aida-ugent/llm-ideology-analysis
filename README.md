# LLM Ideology Analysis

[![Paper](https://img.shields.io/badge/paper-pdf-blue.svg)](https://arxiv.org/abs/2410.18417)
[![Dataset](https://img.shields.io/badge/🤗_dataset-huggingface-yellow.svg)](https://huggingface.co/datasets/ajrogier/llm-ideology-analysis)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This repository contains the code and analysis tools for the paper "Large Language Models Reflect the Ideology of their Creators". We provide a comprehensive framework for analyzing ideological biases in Large Language Models (LLMs) through their evaluations of historical political figures.

## 📊 Dataset

The dataset contains evaluations from 19 different LLMs of 3,991 political figures, with responses in all six UN languages (Arabic, Chinese, English, French, Russian, and Spanish). Access the full dataset on [Hugging Face](https://huggingface.co/datasets/aida-ugent/llm-ideology-analysis).

## 📚 Setup and Usage

### Prerequisites
- Python 3.11 or higher
- Poetry (for dependency management)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aida-ugent/llm-ideology-analysis.git
   cd llm-ideology-analysis
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

### Environment Configuration
1. Copy the environment template:
   ```bash
   cp .env.template .env
   ```

2. Configure the following environment variables in `.env`:

   #### API Keys (required for respective models)
   - `OPENAI_API_KEY`: OpenAI API key
   - `ANTHROPIC_API_KEY`: Anthropic API key
   - `HUGGINGFACE_TOKEN`: Hugging Face token
   - `MISTRAL_API_KEY`: Mistral API key
   - `TOGETHER_API_KEY`: Together API key
   - `PERPLEXITY_API_KEY`: Perplexity API key
   - `GEMINI_API_KEY`: Google Gemini API key

   #### Directory Paths
   - `RESULTS_DIR`: Directory for storing results
   - `NOTEBOOKS_DIR`: Directory containing analysis notebooks
   - `DOCS_DIR`: Directory for documentation
   - `FIGURES_DIR`: Directory for generated figures
   - `CACHE_PATH`: Path for caching results

### Running the Analysis

1. Process questions through the unified API:
   ```bash
   poetry run python src/run_questions_through_unified_api.py
   ```

2. Run the manifesto tagger:
   ```bash
   poetry run python src/run_manifesto_tagger.py
   ```

3. Analyze results using Jupyter notebooks in the `notebooks/` directory:


## 📚 Citation

```bibtex
@misc{buyl2024largelanguagemodelsreflect,
      title={Large Language Models Reflect the Ideology of their Creators}, 
      author={Maarten Buyl and Alexander Rogiers and Sander Noels and Iris Dominguez-Catena and Edith Heiter and Raphael Romero and Iman Johary and Alexandru-Cristian Mara and Jefrey Lijffijt and Tijl De Bie},
      year={2024},
      eprint={2410.18417},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.18417}, 
}
```

## 👥 Team
### Authors
* Maarten Buyl (*‡) - Ghent University, Belgium  
* Alexander Rogiers (†) - Ghent University, Belgium  
* Sander Noels (†) - Ghent University, Belgium
* Guillaume Bied - Ghent University, Belgium
* Iris Dominguez-Catena - Public University of Navarre, Spain  
* Edith Heiter - Ghent University, Belgium  
* Iman Johary - Ghent University, Belgium  
* Alexandru-Cristian Mara - Ghent University, Belgium  
* Raphael Romero - Ghent University, Belgium  
* Jefrey Lijffijt - Ghent University, Belgium  
* Tijl De Bie - Ghent University, Belgium  

\* Corresponding author: maarten.buyl@ugent.be  
† These authors contributed equally to this work  
‡ Lead author

### Affiliations
* **Ghent University**  
  Department of Electronics and Information Systems  
  IDLab  
  Technologiepark-Zwijnaarde 122  
  9052 Ghent, Belgium  

* **Public University of Navarre**  
  Department of Statistics, Computer Science and Mathematics  
  31006 Pamplona, Spain

## 📧 Contact

For questions or issues, please:
1. Open an issue in this repository
2. Contact one of the corresponding authors: maarten.buyl@ugent.be,
   alexander.rogiers@ugent.be or sander.noels@ugent.be
