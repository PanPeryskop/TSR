# TSR - Traffic Sign Recognition

### [Watch a live demonstration of the system here](https://youtu.be/D8gwmzYyI30)
This project is a Traffic sign recognition and text-to-speech system based on machine learning.

## Traffic Sign Recognition

- Utilizes the [YOLO (YOLOv11)](https://github.com/ultralytics/ultralytics) model for real-time traffic sign detection.
- The model is trained on a dataset specified in the `config.yaml` file and saved in ONNX and PyTorch formats.
- Sign detection with tts is implemented in the `tsr.py` script, which uses a camera to capture images and detect traffic signs and uses the ParlerTTS model to generate voice messages based on the detected signs.
- You can see a live preview in `rt_tsr.py`.


## Text-to-Speech

- Uses the ParlerTTS model to generate speech based on detected traffic signs.
- The implementation is found in the `TSR_TTS` class in the `tsr_tts.py` file.

## Model Training

- The process of training the YOLO model is visible in the Jupyter notebook `training.ipynb`.
- Training parameters are configured in the `config.yaml` file.

## Project Structure

- The `models` folder contains saved models in ONNX and PyTorch formats.
- The `data` folder contains training, testing, and validation data.
- The `audio` folder stores generated audio files and temporary sound files.
- Archives:
    - `old_models` folder contains old models.
    - `old_data` contains old datasets (not uploaded to GitHub).

## How It Works

1. **Traffic Sign Detection**: The system captures images from a camera and uses the YOLO model to detect traffic signs in real-time. The detected signs are then classified and labeled.
2. **Text-to-Speech Generation**: Based on the detected signs, the system generates corresponding voice messages using the ParlerTTS model. These messages are then played back to the user.

## Usage

1. **Setup**: Ensure you have all the required dependencies installed. Make sure you have the correct versions of Torch and CUDA installed. You can install them from [PyTorch's official site](https://pytorch.org/get-started/locally/).
2. **Running the Traffic Sign Recognition with tts**:
    ```bash
    python tsr.py
    ```

3. **Running real time preview**:
    ```bash
    python rt_tsr.py
    ```

## Final Model
- The final model is saved in the `models` folder as 'tsrm.onnx' and 'tsrm.pth'.
- It was trained using the YOLOv11 medium model.
- Final model was trained on a dataset of 1060 images of traffic signs/
- The model was trained for 350 epochs.
- The model can recognize 24 classes of Polish traffic signs:

  - **A-1**: Niebezpieczny zakręt w prawo
  - **A-11a**: Próg zwalniający
  - **A-16**: Przejście dla pieszych
  - **A-17**: Uwaga dzieci
  - **A-2**: Niebezpieczny zakręt w lewo
  - **A-30**: Inne niebezpieczeństwo
  - **A-7**: Ustąp pierwszeństwa
  - **B-1**: Zakaz ruchu w obu kierunkach
  - **B-2**: Zakaz wjazdu
  - **B-20**: STOP
  - **B-21**: Zakaz skręcania w lewo
  - **B-22**: Zakaz skręcania w prawo
  - **B-23**: Zakaz zawracania
  - **B-33**: Ograniczenie prędkości
  - **B-36**: Zakaz zatrzymywania się
  - **B-41**: Zakaz ruchu pieszych
  - **C-12**: Rondo
  - **C-2**: Nakaz jazdy w prawo za znakiem
  - **C-5**: Nakaz jazdy prosto
  - **D-1**: Droga z pierwszeństwem
  - **D-18**: Parking
  - **D-3**: Droga jednokierunkowa
  - **D-6**: Przejście dla pieszych
  - **D-6b**: Przejście dla pieszych i droga dla rowerzystów
- You can see the process of training the model in the `training.ipynb` notebook.

## Model Performance

The model's performance was evaluated on a validation dataset. Below are examples of the ground truth labels and the model's predictions:

| Image | Description |
|-------|-------------|
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/val_batch2_labels.jpg" alt="Ground Truth" width="400"/> | *Labels from dataset* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/val_batch2_pred.jpg" alt="Predictions" width="400"/> | *Predictions* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/confusion_matrix_normalized.png" alt="Confusion Matrix" width="400"/> | *Confusion Matrix* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/R_curve.png" alt="R Curve" width="400"/> | *R Curve* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/P_curve.png" alt="P Curve" width="400"/> | *P Curve* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/PR_curve.png" alt="PR Curve" width="400"/> | *PR Curve* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/F1_curve.png" alt="F1 Curve" width="400"/> | *F1 Curve* |
