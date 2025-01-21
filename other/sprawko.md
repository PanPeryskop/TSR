# System Rozpoznawania Znaków Drogowych (TSR) - Szczegółowe sprawozdanie

### [Demonstracja działania systemu](https://youtu.be/D8gwmzYyI30)

## 1. Wprowadzenie

Projekt realizuje kompleksowy system rozpoznawania znaków drogowych (Traffic Sign Recognition - TSR)
wykorzystujący uczenie głębokie oraz syntezę mowy. System działa w czasie rzeczywistym, wykrywając
polskie znaki drogowe i komunikując ich znaczenie poprzez komunikaty głosowe.

## 2. Wykorzystane technologie

### 2.1 Trenowanie Modelu

- Model YOLOv11 (medium)
- PyTorch
- CUDA

### 2.2 Przetwarzanie obrazu

- OpenCV - przechwytywanie i przetwarzanie obrazu oraz wyświetlanie pogdlądu

### 2.3 Synteza mowy

- ParlerTTS - generowanie komunikatów głosowych

## 3. Architektura systemu

### 3.1 Struktura projektu

```
project/
├── models/              # Wytrenowane modele
│   ├── tsrm.onnx       # Model YOLO (ONNX)
│   └── tsrm.pt         # Model YOLO (PyTorch)
├── data/               # Dataset
├── audio/              # Pliki dźwiękowe
└── src/
    ├──
tsr.py
    ├──
rt_tsr.py
    └──
tsr_tts.py
```

### 3.2 Komponenty systemu

1. **Moduł detekcji (tsr.py)**

   - Przechwytywanie obrazu z kamery
   - Detekcja znaków w czasie rzeczywistym
   - Klasyfikacja wykrytych obiektów

2. **Moduł syntezy mowy (tsr_tts.py)**

   - Generowanie komunikatów głosowych
   - Zarządzanie kolejką komunikatów
   - Odtwarzanie dźwięku

3. **Moduł podglądu (rt_tsr.py)**
   - Wizualizacja detekcji

## 4. Model rozpoznawania znaków

### 4.1 Specyfikacja modelu

- Architektura: YOLOv11 medium
- Format: ONNX + PyTorch
- Rozmiar zbioru danych: 1060 obrazów
- Liczba epok: 350
- Liczba klas: 24 polskie znaki drogowe

### 4.2 Obsługiwane klasy znaków

#### Znaki ostrzegawcze (A):

- A-1: Niebezpieczny zakręt w prawo
- A-2: Niebezpieczny zakręt w lewo
- A-7: Ustąp pierwszeństwa
- A-11a: Próg zwalniający
- A-16: Przejście dla pieszych
- A-17: Uwaga dzieci
- A-30: Inne niebezpieczeństwo

#### Znaki zakazu (B):

- B-1: Zakaz ruchu w obu kierunkach
- B-2: Zakaz wjazdu
- B-20: STOP
- B-21: Zakaz skręcania w lewo
- B-22: Zakaz skręcania w prawo
- B-23: Zakaz zawracania
- B-33: Ograniczenie prędkości
- B-36: Zakaz zatrzymywania się
- B-41: Zakaz ruchu pieszych

#### Znaki nakazu (C):

- C-2: Nakaz jazdy w prawo za znakiem
- C-5: Nakaz jazdy prosto
- C-12: Rondo

#### Znaki informacyjne (D):

- D-1: Droga z pierwszeństwem
- D-3: Droga jednokierunkowa
- D-6: Przejście dla pieszych
- D-6b: Przejście dla pieszych i przejazd dla rowerzystów
- D-18: Parking

## 5. Działanie i ewaluacja modelu

Model został poddany szczegółowej ewaluacji na zbiorze walidacyjnym. Poniżej przedstawiono wyniki wraz z interpretacją:

| Wizualizacja                                                                                                                                                      | Opis                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/val_batch2_labels.jpg" alt="Ground Truth" width="400"/>               | **Etykiety rzeczywiste** <br> Przedstawienie prawidłowych etykiet ze zbioru walidacyjnego.                            |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/val_batch2_pred.jpg" alt="Predictions" width="400"/>                  | **Predykcje modelu** <br>                     |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/confusion_matrix_normalized.png" alt="Confusion Matrix" width="400"/> | **Macierz pomyłek** <br> Znormalizowana macierz pokazująca skuteczność klasyfikacji dla poszczególnych klas znaków. |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/R_curve.png" alt="R Curve" width="400"/>                              | **Krzywa Recall** <br> Pokazuje czułość modelu - zdolność do wykrycia wszystkich istotnych obiektów.     |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/P_curve.png" alt="P Curve" width="400"/>                              | **Krzywa Precision** <br> Prezentuje precyzję modelu - dokładność pozytywnych predykcji.              |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/PR_curve.png" alt="PR Curve" width="400"/>                            | **Krzywa Precision-Recall** <br> Kompromis między precyzją a czułością modelu.                  |
| <img src="https://raw.githubusercontent.com/PanPeryskop/TSR/refs/heads/main/runs/detect/val/F1_curve.png" alt="F1 Curve" width="400"/>                            | **Krzywa F1** <br> Średnia harmoniczna precision i recall.                                       |

### 5.1 Wydajność modelu

- **Średnia precyzja (mAP)**: 91%
- **Średni współczynnik F1**: 0.90
- **Średnia czułość (Recall)**: 89%
- **Próg pewności (Confidence threshold)**: 0.75

### 5.2 Analiza wyników

1. Model wykazuje szczególnie wysoką skuteczność w rozpoznawaniu:

   - Znaków STOP
   - Znaków ostrzegawczych
   - Przejść dla pieszych

2. Wyzwania:
   - Znaki w warunkach słabego oświetlenia
   - Częściowo zasłonięte znaki
   - Znaki o podobnej geometrii (np. ograniczenia prędkości)

### 6.0 Jak używać?

### 6.0 Jak używać?

1. **Instalacja zależności**:

   - Upewnij się, że masz zainstalowane odpowiednie wersje PyTorch i CUDA ze strony [PyTorch](https://pytorch.org/get-started/locally/)
   - Zainstaluj wymagane biblioteki:
     ```bash
     pip install ultralytics
     pip install opencv-python
     pip install git+https://github.com/huggingface/parler-tts.git
     pip install simpleaudio
     pip install soundfile
     pip install numpy
     ```

2. **Uruchomienie systemu z syntezą mowy**:

   ```bash
   python tsr.py
   ```

3. **Uruchomienie podglądu w czasie rzeczywistym**:

   ```bash
   python rt_tsr.py
   ```

4. **Uwagi**:
   - Można używać 2 opcji jednocześnie w celu uruchomiena kominikatów audio oraz wizualizacji
   - System domyślnie używa kamery o indeksie 2
   - Próg pewności detekcji ustawiony na 0.75
   - Rozdzielczość przechwytywania: 960x540 px

# [Kod źródłowy dostępny na GitHub](https://github.com/PanPeryskop/TSR)