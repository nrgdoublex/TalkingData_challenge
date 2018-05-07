# TalkingData_challenge

## Challenge description
Given the data of users clicking the ad, predict the probability that the app is downloaded.
### Data description
The training data contains 184903891 rows. Each row is a clicking record with the following features:
* ip: ip address of click.
* app: app id for marketing.
* device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei * mate 7, etc.)
* os: os version id of user mobile phone
* channel: channel id of mobile ad publisher
* click_time: timestamp of click (UTC)
* attributed_time: if user download the app for after clicking an ad, this is the time of the app download
* is_attributed: the target that is to be predicted, indicating the app was downloaded

## Project overview
We have 2 implementations, deep learning and gradient boosting models, for this challenge:
* deep learning model: 5-layer dense-dropout model, with original,time and group features.
* gradient boosting model: used xgboost with original and time features.
