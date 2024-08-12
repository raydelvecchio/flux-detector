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

* Non-Augmented:
    * 0.9401 Test Accuracy
* Augmented:
    * 0.7667 Test Accuracy

I suspect that this result is because the images are 32x32. The inversion and saturation are only obviously visibile in high resolution imagery.

# FLUX Testing
Ideally, we test this on some semi-SOTA realistic image models, not just CIFAKE. So this is what we do!

### Dataset Generation
I created my own dataset for this! Two classes (AI generated vs real) of women speaking on stage (inspired by that original [viral image](https://x.com/AngryTomtweets/status/1822203767728591350)). In retrospect, I probably should have picked something that would be more consistent between Flux's generation and google images. Oh well. The real ones are scraped from Google Images, and
the AI generated ones use [Replicate's FLUX Schneill](https://replicate.com/black-forest-labs/flux-schnell) model. The AI generated images were randomly seeded with a base prompt and some randomly selected modifier for a broader distribution. Yes, I know it's not the SOTA one for realize, but we're working efficiently with CNNs here. Give me a break. But yes, we should be using the most real, and I would if it wasn't like $0.03 a generation request.
