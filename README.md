__Face Recognition with MobileNetV2 & MTCNN__

This project implements a simple face recognition system using:

MobileNetV2 for feature extraction
MTCNN for face detection
Python 3.9+ in a clean environment

__Environment Setup__

1. Create and Activate a Clean Environment
With Conda (recommended):

conda create -n facerec python=3.9
conda activate facerec

Or with venv (if you prefer):

python3 -m venv facerec-env
source facerec-env/bin/activate  # On Windows: .\facerec-env\Scripts\activate

2. Install Required Packages
On Windows/Linux/Intel Mac:

pip install --upgrade pip
pip install tensorflow scikit-learn mtcnn pillow protobuf==3.20.3 numpy==1.23.5 h5py==3.10.0 opencv-python
Use tensorflow (not tensorflow-macos or tensorflow-metal).

On Apple Silicon Mac:

pip install tensorflow-macos tensorflow-metal scikit-learn mtcnn pillow protobuf==3.20.3 numpy==1.23.5 h5py==3.10.0 opencv-python

3. Dataset Structure

dataset/
  person1/
    img1.jpg
    img2.jpg
  person2/
    img1.jpg
    img2.jpg
test.jpg
Each folder inside dataset/ is a person/class.
Images should be clear, standard .jpg .jpeg or .png files with visible faces.

4. Running the Script

python /path/to/your/face_detection.py

__Troubleshooting__

If you see errors about libprotobuf, numpy, or TypeError: bases must be types:

Make sure you are using a clean environment.
Do not mix conda and pip installations for core packages.
Use only the package versions listed above.
If you see errors about missing packages:
Re-run the pip install ... command above.

This setup works on Windows, Linux, and Intel Macs.
For best results, use clear, single-face images.
If you need to add more people, just add more folders and images to dataset/.
