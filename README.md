# AI Image Detector
Can we train a SMALL model to consistenyl recognize AI vs real imagery? Let's find out!

# Methodology
* AI image generators are good at generating visually accurate images in a *normal RGB space*
* But what about ABNORMAL color spaces? Do we see any breakdown in their capability?
* By modifying the image and applying some augmentations, we can reveal many imperfections in AI generated images vs real ones
* For example, if we invert the colors of the image, then massively increase the saturation, we can see how edges, hair, and other image components start to break down in an unnatural manner
    * You can see some examples of this in `./scripts`
* Theoretically, if we can visually identify it pretty easy, we can also train a model to do so!
* Originally saw this technique for visually identifying fakes via saturation increase [here](https://www.linkedin.com/posts/debarghyadas_hack-to-tell-if-an-image-is-ai-generated-activity-7228431618380554240-iP7b?utm_source=share&utm_medium=member_desktop)!

# CIFAKE Testing
We test on the [CIFAKE dataset](https://ieeexplore.ieee.org/abstract/document/10409290) as a toy example in the `./cifake` folder.

### Download the Dataset
We're using the CIFAKE dataset, located on [Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images?resource=download).

I've exposed a script `download.py` to do this. To run `download.py`, you must download your `kaggle.json`, containing
your username and key, then place it in `/Users/<USERNAME>/.kaggle` (if on UNIX).

### Results
Here's the accuracy results for the standard CNN vs Inverted / Saturated CNN:

* Params:
    * Current params in the model; don't see myself changing or updating soon
    * Batch size 32
    * 5 Epochs
* Non-Augmented Test Accuracy: 0.9401
* Augmented Test Accuracy: 0.7667

I suspect that this result is because the images are 32x32. The inversion and saturation are only obviously visibile in high resolution imagery.

# FLUX Testing
Ideally, we test this on some semi-SOTA realistic image models, not just CIFAKE. So this is what we do!

### Dataset Generation
I created my own dataset for this! Two classes (AI generated vs real) of women speaking on stage (inspired by that original [viral image](https://x.com/AngryTomtweets/status/1822203767728591350)). In retrospect, I probably should have picked something that would be more consistent between Flux's generation and google images. Oh well. The real ones are scraped from Google Images, and
the AI generated ones use [Replicate's FLUX Schneill](https://replicate.com/black-forest-labs/flux-schnell) model. The AI generated images were randomly seeded with a base prompt and some randomly selected modifier for a broader distribution. Yes, I know it's not the SOTA one for realize, but we're working efficiently with CNNs here. Give me a break. But yes, we should be using the most real, and I would if it wasn't like $0.03 a generation request.

### Results
Splitting up results here by time so I can refine this process over time!

* 12:10 AM Aug 12 2024:
    * Params:
        * Model Architecture:
            ``` 
            FakeDetectorCNN(
            (conv): Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3))
                (1): ReLU()
                (2): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
                (3): Conv2d(32, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
                (4): ReLU()
                (5): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
                (6): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (7): ReLU()
                (8): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (9): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (10): ReLU()
                (11): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
                (12): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (13): ReLU()
                (14): AdaptiveAvgPool2d(output_size=(1, 1))
            )
            (fc): Sequential(
                (0): Flatten(start_dim=1, end_dim=-1)
                (1): Linear(in_features=512, out_features=1024, bias=True)
                (2): ReLU()
                (3): Dropout(p=0.5, inplace=False)
                (4): Linear(in_features=1024, out_features=256, bias=True)
                (5): ReLU()
                (6): Dropout(p=0.5, inplace=False)
                (7): Linear(in_features=256, out_features=1, bias=True)
                (8): Sigmoid()
            )
            (criterion): BCELoss()
            )
            ```
        * 16 Batch Size
        * 5 Epochs
        * 500x Saturation Factor
    * Non-Augmented Test Accuracy: 0.8032
        * Epoch 1 Val Accuracy: 0.5789
        * Epoch 2 Val Accuracy: 0.6842
        * Epoch 3 Val Accuracy: 0.7632
        * Epoch 4 Val Accuracy: 0.8053
        * Epoch 5 Val Accuracy: 0.8105
    * Augmented Test Accuracy: didn't get that far
        * Epoch 1 Val Accuracy: 0.5421
        * Epoch 2 Val Accuracy: 0.5947
* 12:45 AM Aug 12 2024:
    * Params:
        * Same architecture as above
        * 16 Batch Size
        * 5 Epochs
        * 8x Saturation Factor
    * Augmented Test Accuracy: 0.8085
        * Epoch 1 Val Accuracy: 0.4579
        * Epoch 2 Val Accuracy: 0.5421
        * Epoch 3 Val Accuracy: 0.6263
        * Epoch 4 Val Accuracy: 0.7632
        * Epoch 5 Val Accuracy: 0.7737
    
### Conclusions
* It appears that both techniques, through many epochs, are about the same
* I wonder how much of this success is overfitting
* It seems that the augmentations, over many epochs, improve faster than non-augmentations
* Qualitatively, looking at the training samples, the high saturation rate seems to "hide" imperfections in the AI generated imagery
    * This could, in the beginning, make it harder to identify fake vs not fake, contributing to early low accuracy
    * But then, later on, with more epochs or a powerful enough model, this forces the model to look at the inverse / saturation / other artifacts to make a decision perhaps? Which means it could be more effective?
* I think that more data, or more homogenous data, is required to actually draw a conclusion from this!

# TODOs
1. Collect more imagery, or imagery of more homogenous things
2. Print or display what a high saturation image looks like during training
3. Fuck with hyperparameters
