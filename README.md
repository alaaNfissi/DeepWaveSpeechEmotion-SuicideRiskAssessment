<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />

<div align="center">
  <a href="https://github.com/alaaNfissi/DeepWaveSpeechEmotion-SuicideRiskAssessment">
    <img src="figures/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Deep Learning Multiresolution Wavelet Transform for Speech Emotional State Assessment of High-Risk Suicide Callers</h3>

  <p align="center">
    This paper has been submitted for publication in The 11th IAPR TC3 Workshop on Artificial Neural Networks in Pattern Recognition.
    <br />
   </p>
   <!-- <a href="https://github.com/alaaNfissi/DeepWaveSpeechEmotion-SuicideRiskAssessment"><strong>Explore the docs »</strong></a> -->
</div>
   

  
<div align="center">

[![view - Documentation](https://img.shields.io/badge/view-Documentation-blue?style=for-the-badge)](https://github.com/alaaNfissi/DeepWaveSpeechEmotion-SuicideRiskAssessment/#readme "Go to project documentation")

</div>  


<div align="center">
    <p align="center">
    ·
    <a href="https://github.com/alaaNfissi/DeepWaveSpeechEmotion-SuicideRiskAssessment/issues">Report Bug</a>
    ·
    <a href="https://github.com/alaaNfissi/DeepWaveSpeechEmotion-SuicideRiskAssessment/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#abstract">Abstract</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#getting-the-code">Getting the code</a></li>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#reproducing-the-results">Reproducing the results</a></li>
      </ul>
    </li>
    <li>
      <a href="#results">Results</a>
      <ul>
        <li><a href="#on-nspl-crise-dataset">On NSPL-CRISE dataset</a></li>
        <li><a href="#on-iemocap-dataset">On IEMOCAP dataset</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABSTRACT -->
## Abstract

<p align="justify"> This study introduces an end-to-end (E2E) deep learning framework for speech emotion recognition (SER), aimed at enhancing the early detection of suicidal ideation and reducing suicide risks. Despite progress, SER still faces challenges like system complexity, feature distinctiveness, and noise interference. Our approach uses a learnable architecture for fast discrete wavelet transform (FDWT) multi-resolution analysis, directly extracting features from raw speech waveforms. It combines a 1D dilated convolutional neural network (1D dilated CNN) with spatial attention (SA) and bidirectional gated recurrent units (Bi-GRU) with temporal attention (TA) to capture spatial and temporal characteristics. The framework handles variable-length speech without segmentation, simplifying preprocessing. It introduces a learnable architecture for wavelet bases and coefficient denoising with a learnable asymmetric hard thresholding (LAHT) activation function, enhancing noise resilience and feature distinctiveness. We validate our model on the NSPL-CRISE dataset, which includes recordings from individuals with psychological challenges and potential suicidal thoughts. Our results show significant performance improvements over state-of-the-art SER methods, demonstrating the effectiveness of our multi-resolution deep learning framework in identifying emotional states related to suicide risk.</p>
<div align="center">
  
![model-architecture][model-architecture]
  
*SigWavNet General Architecture*
  
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
* ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
* ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
* ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
* ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
* ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
* ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started
<p align="justify">
To ensure consistency and compatibility across our datasets, we first convert all audio signals to a uniform 16 KHz sampling rate and mono-channel format. We then divide each dataset into two primary subsets: 90% for training and validation purposes, and the remaining 10% designated for testing as unseen data. For the training and validation segments, we implement a 10-fold cross-validation method. This partitioning and the allocation within the cross-validation folds leverage stratified random sampling, a method that organizes the dataset into homogenous strata based on emotional categories. Unlike basic random sampling, this approach guarantees a proportional representation of each class, leading to a more equitable and representative dataset division.</p>

<p align="justify">
In the quest to identify optimal hyperparameters for our model, we utilize a grid search strategy. Hyperparameter tuning can be approached in several ways, including the use of scheduling algorithms. These schedulers can efficiently manage trials by early termination of less promising ones, as well as pausing, duplicating, or modifying the hyperparameters of ongoing trials. For its effectiveness and performance, we have selected the Asynchronous Successive Halving Algorithm (ASHA) as our optimization technique.
The data preprocessing used in this study is provided in the `Data_exploration` folder.  
</p>

### Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/alaaNfissi/DeepWaveSpeechEmotion-SuicideRiskAssessment.git

or [download a zip archive](https://github.com/alaaNfissi/DeepWaveSpeechEmotion-SuicideRiskAssessment/archive/refs/heads/main.zip).

### Dependencies

<p align="justify">
You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.txt`.
We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).
Run the following command to create an `ser-env` environment to create a separate environment:
  
```sh 
    conda create --name ser-env
```

Activate the environment, this will enable it for your current terminal session. Any subsequent commands will use software that is installed in the environment:

```sh 
    conda activate ser-env
 ```

Use Pip to install packages to the Anaconda Environment:

```sh 
    conda install pip
```

Install all required dependencies in it:

```sh
    pip install -r requirements.txt
```
  
</p>

### Reproducing the results

<p align="justify">

1. First, you need to download IEMOCAP dataset:
  * [IEMOCAP official website](https://sail.usc.edu/iemocap/)
  
2. To be able to explore the data you need to execute the Jupyter Notebook that prepares the `csv` files needed for the experiments.
To do this, you must first start the notebook server by going into the
repository top level and running:
```sh 
    jupyter notebook
```
This will start the server and open your default web browser to the Jupyter
interface. On the page, go into the `Data_exploration` folder and select the
`data_exploration.ipynb` notebook to view/run. Make sure to specify the correct dataset paths on your machine as described in the notebook.
The notebook is divided into cells (some have text while others have code).
Each cell can be executed using `Shift + Enter`.
Executing text cells does nothing and executing code cells runs the code and produces its output.
To execute the whole notebook, run all cells in order.

3. After generating the needed `csv` file `IEMOCAP_dataset.csv`, go to your terminal where the `ser-env` environment was
  activated go to the project folder and run the python script `main.py` as follows:

```sh  
python main.py
``` 

</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results

### On NSPL-CRISE dataset
<p align="justify"> 
The evaluation of SigWavNet on the NSPL-CRISE dataset provides a comprehensive analysis of its ability to distinguish between various emotional states, as demonstrated by its commendable precision, recall, and F1-score metrics for different emotions. The model accurately identifies 'Happy' emotions 73.68% of the time, demonstrating effective detection of positive cues. It performs well in recognizing 'Sad' with a 73.33% precision and detects 'Neutral' emotions with a 76.92% accuracy rate, indicating strength in handling both pronounced and subtle emotional states. The confusion matrix shows some misclassifications, such as 'Angry' being confused with 'Sad' or 'Neutral', but these occur at lower percentages, suggesting a good overall grasp of emotional nuances. The model's 61.36% accuracy in distinguishing 'FCW' while occasionally confusing it with 'Sad' highlights areas for potential refinement.</p>

SigWavNet confusion matrix on NSPL-CRISE           | 
:-----------------------------------------------------------------:|
![sigwavnet_cfm_emodb](figures/nspl_crise_cfm_1.png)  |


### On IEMOCAP dataset
<p align="justify"> 
The trials showcase the proficiency of the SigWavNet model in recognizing diverse emotional expressions from the IEMOCAP dataset. This model achieves notable accuracy in distinguishing between various emotions, as indicated by its performance metrics—precision, recall, and F1-score—across different emotional categories. Specifically, SigWavNet performs exceptionally well in identifying 'Neutral' emotions, achieving a high precision rate of 97% and a recall rate of 93% (refer to the paper). This underscores the model's strength in accurately pinpointing this particular emotional state. The confusion matrix in SigWavNet confusion matrix figure describes class-wise test results on IEMOCAP. 
</p>

SigWavNet confusion matrix on IEMOCAP            | 
:-----------------------------------------------------------------:|
![sigwavnet_cfm_iemocap](figures/iemocap_cfm.png)  |


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<p align="center">
  
_For more detailed experiments and results you can read the paper._
</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Alaa Nfissi - [@LinkedIn](https://www.linkedin.com/in/alaa-nfissi/) - alaa.nfissi@mail.concordia.ca

Github Link: [https://github.com/alaaNfissi/DeepWaveSpeechEmotion-SuicideRiskAssessment](https://github.com/alaaNfissi/DeepWaveSpeechEmotion-SuicideRiskAssessment)

<p align="right">(<a href="#readme-top">back to top</a>)</p>




<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[model-architecture]: figures/SigWavNet_ANNPR_small.png


[anaconda.com]: https://anaconda.org/conda-forge/mlconjug/badges/version.svg
[anaconda-url]: https://anaconda.org/conda-forge/mlconjug

[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
