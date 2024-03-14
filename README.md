# Wax Apple Hyperspectral-Image Open Dataset
[DATASET-LINK]
## **Materials and Data preparation** 
In the study, **136 wax apple samples** were collected from Meishan, Fengshan, Liouguei, and Jiadong. These samples then cut into slices **(1034 slices in total)** for HSIs data collection by the **coaxial heterogeneous HSI system** [1]. The raw hyperspectral image (HSI) is **three-dimensional data $(W, L, Λ)$**, where $W$ is image width, $L$ is image length, $Λ$ is the number of spectral bands, which has **1367 bands from 400 to 1700 nm**.<br>
Each slice’s HSI were calibrated with spatial calibration, white/dark light calibration, and Savitzky-Golay Filtering, then each HSIs cube was randomly **cropped into size 20×20**. The **3-D cubes dataset** would be used to train **2D-CNN** models, for **FNN** model the 3-D datasets were averaged to **array-like datasets** by their width and length.<br>
On the other hand, each wax apple slice was squeezed to extract juice sample, and the **ground truth Brix value (label)** was measured by a commercial refractometer **ATAGO PAL-1**.

![image](https://hackmd.io/_uploads/SJawT1cpp.png )

After the datasets were prepared, they were randomly sampled into **"training", "validation", and "test"** for modeling. The statistic info of each set is shown in below table. (we randomly discard some samples to make dataset balanced.)
**Training set : 621 slices, validation set : 209 slices, test set : 204 slices.**

![圖片5](https://hackmd.io/_uploads/r15PNx56T.png )


<br>

Below is the file structure and code snippet for loading HIS data samples and corresponding labels.
```
├── Hyper
│   ├── 3d_data
│   │   └── 400_1700
│   │       ├── data_test.pkl
│   │       ├── data_train.pkl
│   │       ├── data_val.pkl
│   │       ├── label_test.pkl
│   │       ├── label_train.pkl
│   │       └── label_val.pkl
└─  └──
```

```python=
import numpy as np
import pickle as pkl

PATH = 'D:/wax_apple/Hyper/3d_data/400_1700/' # change to your path here
x_train = np.load(PATH + 'data_train.pkl', 'rb', allow_pickle = True).astype('float32')
x_val = np.load(PATH + 'data_val.pkl', allow_pickle = True).astype('float32')
x_test = np.load(PATH + 'data_test.pkl', allow_pickle = True).astype('float32')


with open(PATH + 'label_train.pkl', 'rb') as f:
    y_train = np.array(pkl.load(f))
with open(PATH + 'label_val.pkl', 'rb') as f:
    y_val = np.array(pkl.load(f))
with open(PATH + 'label_test.pkl', 'rb') as f:
    y_test = np.array(pkl.load(f))

#shape of x (HSI Data)            #shape of y (Brix Data) 
#x_train : (621, 20, 20, 1367)    y_train : (621,)
#x_val   : (209, 20, 20, 1367)    y_val   : (209,)
#x_test  : (204, 20, 20, 1367)    y_test  : (204,)
```

<br>

## Convert Hyperspectral Data to Multispectral Data
In the coaxial heterogeneous HSI system, the VIS spectrometer has a spectral resolution of ~0.5 nm in the wavelength ranging from 400 to 1000 nm; the SWIR spectrometer has a spectral resolution of ~2.5 nm in the wavelength ranging from 900 to 1700 nm.

**The hyperspectral data were converted to multispectral data, $R_M(x, y, λc)$ using following fomula:**

$$
R_M\ (x,\ y,\lambda_c\ ) = {\frac{1}{\lambda_b-\lambda_a} [\sum_{i=a}^{b} R_{SGF}\ (x,y,\lambda_i\ )\times(\lambda_{i+1}-\lambda_i\ )\ ],[λ_a,λ_b ]∈[λ_c-\frac{w}{2},λ_c+\frac{w}{2}]}
$$ 

**where $R_M(x, y, λc)$ is the mean of the integrated intensity of the central band, $λc$; the central band $λc$ ranges from 400 to 1700 nm with an interval of bandwidth, $w$**. 


<div class="content">
    <img src="https://hackmd.io/_uploads/S1VP5_2ap.png" width="400" height="150" style="float:left;margin:0 160px 12px" >
</div>

A total of six sets of spectral data were converted from hyperspectral data according to the to the **six bandwidths, W, of ± 2.5 nm, ± 5 nm, ± 7.5 nm, ± 10 nm, ± 12.5 nm, and ± 15 nm.**

The code snippet to convert Hyperspectral image data to multispectral image data is shown below: 

```python=
import numpy as np

# main function to convert hyperspectal data to multispectral data
def hyper2multi(x, w, hyperband):
    # produce Central band 
    start_band = 400
    end_band = 1700
    central_band = np.arange(start_band+w, end_band, step=w*2) # w*2 = interval

    first = True
    hyperband = hyperband.tolist()
    for cb in central_band:
        start_index = [hyperband.index(band) for band in hyperband if band >= (cb-w)][0] 
        end_index = [hyperband.index(band) for band in hyperband if band < (cb+w+1)][-1]

        if hyperband[end_index] >= hyperband[-1]: # if end_band >= last index in hyperband (1700.075 nm)
            weights = [hyperband[i+1]-hyperband[i] for i in range(start_index, end_index)]
            weights[0] =  weights[0]/2
            weights[-1] =  weights[-1]/2
            temp = np.expand_dims(np.average(x[:,:, :, start_index:end_index], axis=-1, weights=weights), axis=-1)
        else:
            weights = [hyperband[i+1]-hyperband[i] for i in range(start_index, end_index+1)]
            weights[0] =  weights[0]/2
            weights[-1] =  weights[-1]/2
            temp = np.expand_dims(np.average(x[:,:, :, start_index:end_index+1], axis=-1, weights=weights), axis=-1)


        if(first):
            multidata = temp
            first = False
        else:
            multidata = np.concatenate((multidata, temp), axis=-1)
            
    return multidata
```

**Use hyper2multi function to convert train, val, test dataset.**

```python=
import pickle as pkl

# change bandwidth here, w = +- bandwidth (nm)
w = 15 #in the paper w = 2.5, 5, 7.5, 10, 12.5, 15

PATH = YOUR_PATH_TO_hyper_band_list_400-1700.pkl
with open( PATH +"hyper_band_list_400-1700.pkl","rb") as bandfile:
    hyperband = pkl.load(bandfile)

x_train_multi = hyper2multi(x_train, w, hyperband)
x_val_multi = hyper2multi(x_val, w, hyperband)
x_test_multi = hyper2multi(x_test, w, hyperband)

#shape of x (MSI Data)            
#x_train_multi : (621, 20, 20, Band numbers)    
#x_val_multi   : (209, 20, 20, Band numbers)   
#x_test_multi  : (204, 20, 20, Band numbers)
```

<br>

### References
1.Yu-Hsiang Tsai, Yung-Jhe Yan, Yi-Sheng Li, Chao-Hsin Chang, Chi-Cho Huang, Tzung-Cheng Chen, Shiou-Gwo Lin, and Mang Ou-Yang*, “Development and verification of the coaxial heterogeneous hyperspectral imaging system,” Review of Scientific Instruments, Vol.93, pp.063105-1-17, Jun. 2022. DOI: [10.1109/I2MTC.2019.8826836](https://ieeexplore.ieee.org/document/8826836)

2.Chih-Jung Chen, Yung-Jhe Yan, Chi-Cho Huang, Jen-Tzung Chien, Chang-Ting Chu, Je-Wei Jang, Tzung-Cheng Chen, Shiou-Gwo Lin, Ruei-Siang Shih and Mang Ou-Yang*, “Sugariness Prediction of Syzygium samarangense using Convolutional Learning of Hyperspectral Images,” Scientific Reports, 12:2774, 17 Feb. 2022. DOI:[10.1038/s41598-022-06679-6](https://doi.org/10.1038/s41598-022-06679-6)

3.Yung-Jhe Yan, Weng-Keong Wong, Chih-Jung Chen, Chi-Cho Huang, Jen‑Tzung Chien and Mang Ou-Yang*, “Hyperspectral signature-band extraction and learning: an example of sugar content prediction of Syzygium samarangense,”  Scientific Reports, 13:15100, 12 Sep. 2023. DOI: [10.1038/s41598-022-06679-6](https://doi.org/10.1038/s41598-023-41603-6)

### Acknowledgements
This work is supported by grants from Agricultural Research Institute, Council of Agriculture, Executive of Yuan, ROC, and by the National Science and Technology Council, Taiwan , and National Yang Ming Chiao Tung University. This work contributed by following authors: Chih-Jung Chen, Yung-Jhe Yan, Weng-Keong Wong, Chih-Jung Chen, Chi-Cho Huang, Jen-Tzung Chien, Chang-Ting Chu, Je-Wei Jang, Tzung-Cheng Chen, Shiou-Gwo Lin, Ruei-Siang Shih and Mang Ou-Yang
