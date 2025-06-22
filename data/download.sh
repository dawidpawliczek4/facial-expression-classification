#!/bin/bash
curl -L -o ./downloaded_data.zip\
  https://www.kaggle.com/api/v1/datasets/download/astraszab/facial-expression-dataset-image-folders-fer2013

unzip -o downloaded_data.zip -d ./downloaded_data > /dev/null 2>&1
rm downloaded_data.zip > /dev/null 2>&1