PURPOSE:
To learn the fundamentals of Supervised Machine Learning,
specifically Convolutional Neural Networks (CNNs), and to gain familiarity
with notable Machine Learning libraries in Python.

DESCRIPTION:
A web scraper was implemented to create a dataset consisting of 5000 images
of cats and 5000 images of dogs. The images were downloaded from
https://unsplash.com/images/animals/dog and https://unsplash.com/images/animals/cat,
then saved directly to Google Drive. The dataset was split into training (80%),
validation (10%), and testing (10%) sets. The images were downscaled to 125 x 125 pixels
to improve accuracy and efficiency. A deep learning CNN was then developed and optimized
to classify these images. Various techniques, such as data preprocessing, hyperparameter
tuning, and model architecture refinement, were used to evaluate the performance of the
model. The standard accuracy of the preliminary model was ~57%, which was improved to
~81%. Other notable metrics, such as Precision, Recall and F1 score, were also implemented.

PACKAGES:
Keras
Sklearn (scikit-learn)
BeautifulSoup

