# Sistema di Face Detection per una fotocamera digitale

![selfie_oscar](https://github.com/VirginioC/face-detection-fotocamera-digitale/blob/main/selfie_oscar.png)

## Descrizione e obiettivi del progetto
Questo progetto è stato realizzato durante il Master in Data Science di Profession AI utilizzando il linguaggio **Python** e l'ambiente **Google Colab**. L'azienda fittizia ProCam S.p.A. sta sviluppando una nuova fotocamera digitale compatta, pensata per i giovani appassionati di fotografia. L'obiettivo del progetto è implementare un sistema di rilevamento volti per ottimizzare le impostazioni della fotocamera durante i selfie. Il dataset non viene fornito, deve essere quindi cercato in rete o costruito.

Il sistema dovrà:
- Identificare i volti nelle immagini.
- Restituire le coordinate dei bounding boxes dei volti rilevati.
- Restituire una lista vuota se non vengono trovati volti.

Il modello viene sviluppato da zero utilizzando **scikit-learn** senza l'uso di modelli pre-addestrati, documentando le scelte implementate e ottimizzando l'uso delle risorse di calcolo.

## Dataset
Si sceglie di prendere in considerazione due differenti dataset, disponibili online su Kaggle, uno per le immagini contenenti solo volti ed uno per le immagini contenenti "non volti" che sono rispettivamente:

1. **`FaceScrub`**: Contiene volti di 530 celebrità. Per ottimizzare le risorse di calcolo si selezionano casualmente il 50 % di immagini di attori e il 50 % di attrici dal dataset ottenendo: **20966 immagini di volti** in formato **jpeg**, con 3 canali **RGB** e di dimensioni variabili (da **103x103 pixel** a **2857x2857 pixel**).
   
2. **`Self-Taught Learning 10 (STL-10)`**: Contiene 100000 immagini non etichettate e 13000 immagini etichettate di 10 classi di oggetti (come uccelli, gatti, camion). Si considerano in questo caso solo le immagini non etichettate ottenendo: **100000 immagini di non volti** in formato **jpeg** (originariamente in png, covertite per uniformità con FaceScrub), con 3 canali **RGB** e di dimensioni **96x96 pixel**.

Per scaricare i due dataset nel notebook, è necessario configurare le credenziali API di Kaggle:
```python
kaggle_creds = {'KAGGLE_USERNAME': 'your_username',
                'KAGGLE_KEY': 'your_api_key'}

os.environ['KAGGLE_USERNAME'] = kaggle_creds['KAGGLE_USERNAME']
os.environ['KAGGLE_KEY'] = kaggle_creds['KAGGLE_KEY']
```

## Fasi principali del progetto
1. Costruzione del dataset:
   -  Preprocessing dei due dataset:
      - Ridimensionamento di tutte le immagini a 64x64 pixel (dimensione adatta per Face Detection).
      - Conversione in scala di grigi (alleggerimento del carico computazionale).
      - Selezione random di 20966 immagini nel dataset di non volti per bilanciare il numero dei volti.
   - Unione in un **unico dataset** costituito da **41932 immagini di volti e non volti**.
   - Creazione dell'**array del target** etichettando con 1 i volti e con 0 i non volti.

2. Face Classification:
   - Estrazione delle **HOG features** per ciascuna immagine e splitting del dataset in train set (80 %) e test set (20 %).
   - Addestramento di un modello di **SVM** (kernel gaussiano scelto con grid search) per la classificazione binaria.
   - Valutazione del modello: prestazioni quasi perfette con un'**accuracy del 99.93 %** sul test set (6 errori su 8387 immagini).

3. Face Detection:
   - Per ottenere le coordinate dei volti si utilizzano in combinazione due tecniche:
      - **Image Pyramids**: ridimensionamento progressivo delle immagini per rilevare volti di diverse dimensioni.
      - **Sliding Windows**: scansione dell'immagine con finestre mobili per individuare aree di interesse.
   - Si disegnano poi i **bounding boxes** con una funzione specifica e si utilizza un'ulteriore tecnica per migliorare il riconoscimento:
      - **Non Maximum Suppression**: selezione delle bounding boxes più accurate, sopprimendo quelle che rappresentano duplicazioni o sovrapposizioni.

4. Test del prototipo di Face Detection su immagini varie:
   - Viene creata la funzione `face_detection` che racchiude tutte le precedenti operazioni per il riconoscimento volti prendendo come unico input il percorso dell'immagine da esaminare.
   - Si testa il prototipo su immagini con differenti caratteristiche per verificare la bontà del sistema in diverse situazioni: una sola persona, due o tre persone, gruppi di persone, immagini in scala di grigi, dimensioni in pixels e formati differenti.

## Tecnologie utilizzate
- **Linguaggio**: Python
- **Ambiente di sviluppo**: Google Colab (Jupyter Notebook)
- **Librerie principali**:
   - `numpy`
   - `opencv (cv2)`
   - `matplotlib`
   - `scikit-image`
   - `scikit-learn`
   - `joblib`
   - `imutils`

## Utilizzo  
1. Scarica o clona il repository.
2. Apri il file `face_detection_fotocamera_digitale.ipynb` su Google Colab o altri ambienti compatibili con Jupyter Notebook.
3. Esegui il codice passo-passo.

## Fonti e riferimenti
- [FaceScrub - NUS](https://vintage.winklerbros.net/facescrub.html)
- [FaceScrub - Kaggle](https://www.kaggle.com/datasets/rajnishe/facescrub-full/data)
- [STL-10 - Papers With Code](https://paperswithcode.com/dataset/stl-10)
- [STL-10 - Kaggle](https://www.kaggle.com/datasets/jessicali9530/stl10)
- [Histogram of Oriented Gradients (HOG) - Scikit-Image](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html)
- [Image Pyramids - PyImageSearch](https://pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/)
- [Sliding Windows - PyImageSearch](https://pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)
- [Non Maximum Suppression - PyImageSearch](https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)

## Autore
[Virginio Cocciaglia](https://github.com/VirginioC)

---
