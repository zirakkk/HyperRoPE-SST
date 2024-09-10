## **Hyperspectral Image Classification with SQSForme**r

**This project implements a novel approach for hyper**spectral image (HSI) classification using the SQSF**ormer model.**

**## Setup**

**1.** Clone the repository:

**   ```**

**   git clone https://github.com/your-username/hsi-**classification.git

**   cd hsi-classification**

**   ```**

**2.** Create a conda environment with required packages and activate it:

**   ```**

**   conda env create -f environment.yml**

**   conda activate plasticseg**

**   ```**

**## Usage**

**To run an experiment:**

python main.py --config sqsformer_copy.json

You can modify the configuration files in the `configs` directory to adjust the model parameters and dataset settings.

## Project Structure

- `configs/`: Configuration files for experiments
- `data/`: Data loading and preprocessing
- `models/`: Model architecture definitions
- `utils/`: Utility functions, evaluation metrics, and training helpers
- `main.py`: Main script to run experiments

## Results

The results of each experiment will be saved in the `results` directory, including evaluation metrics and model parameters.

## Citation

If you use this code in your research, please cite our paper: @article{your-paper, title={Your Paper Title}, author={Your Name}, journal={Journal Name}, year={2023}}

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
