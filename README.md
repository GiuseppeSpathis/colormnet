# Colorment Fork: Fine-tuning for Enhanced Light Perception on Reflective Surfaces

This repository is a fork of the original [Colorment](https://github.com/yyang181/colormnet) project. The primary goal of this work is to fine-tune the model to improve its perception and rendering of light, particularly how it interacts with reflective surfaces like aluminum under diverse environmental conditions.

## Custom Dataset for Fine-tuning

To target the specific challenge of light perception on reflective materials, a small, specialized dataset was created.

* **Content:** The dataset consists of approximately 40 videos. Each video features a static object containing a distinct aluminum section. Aluminum was chosen for its high sensitivity to variations in light.
* **Variations:** The videos capture the object under a range of conditions:
    * **Color Contexts:** Warm lighting, cool lighting, and black & white environments.
    * **Luminosity:** Brightly lit scenes, dimly lit scenes, and transitions between bright and dim conditions.
* **Objective:** This dataset is designed to train the model to better understand and reproduce the subtle and sometimes dramatic effects of changing light and color temperature on a reflective surface.
* **Download:** The dataset is available for download here:
    [Download Custom Video Dataset](https://liveunibo-my.sharepoint.com/:u:/g/personal/giuseppe_spathis_studio_unibo_it/Eev3GPxlMJpCuPoKQQ3CHdsBpysrXeM5c9C3-Ycl80oruw?e=Jt6Hq7)

## Fine-tuning Process

The original Colorment model was fine-tuned using the custom dataset described above.

* **Training Duration:** The fine-tuning process was carried out for approximately 100 epochs.
* **Goal:** The aim was to adapt the pre-trained model's weights to better specialize in interpreting and rendering the complex interplay of light on the aluminum surface across the different scenarios present in our dataset.

## Evaluation and Results

The performance of the fine-tuned model was assessed through both qualitative and quantitative analysis.

### Qualitative analysis Video Psycho

<table>
  <tr>
    <td align="center">
      <strong>Psycho (gray scale)</strong><br>
      <img src="https://github.com/GiuseppeSpathis/colormnet/blob/main/psycho.gif" alt="Psycho B&N Loop" width="300">
    </td>
    <td align="center">
      <strong>Psycho Colored</strong><br>
      <img src="https://github.com/GiuseppeSpathis/colormnet/blob/main/psychoColored.gif" alt="Psycho Colored Loop" width="300">
    </td>
    <td align="center">
      <strong>Psycho Post-Tuned</strong><br>
      <img src="https://github.com/GiuseppeSpathis/colormnet/blob/main/psychoColoredPostTuning.gif" alt="Psycho Post-Tuned Loop" width="300">
    </td>
  </tr>
</table>

### Quantitative Analysis
* **Metric:** The Structural Similarity Index (SSIM) was used to quantitatively measure the similarity between the model's output and ground truth frames.
* **Results:** The fine-tuning resulted in a measurable improvement in the model's performance according to the SSIM metric. An average increase of **0.04** in the SSIM score was observed post-tuning.
* **Detailed Report:** For a detailed breakdown of the quantitative results across the dataset, please refer to the spreadsheet: `ssim_results4000it.xlsx`.


