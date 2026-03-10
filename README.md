# Detecting Insurance Fraud: My Data Analyst Internship @ Eurolife FFH 

## The Backstory
Between January and March 2024, I had the chance to step into the corporate world as a **Data Analyst Intern** at **Eurolife FFH (Fairfax Group)**. I was placed in the General Insurance department, and my task was to help tackle one of the most expensive problems in the industry: **Automobile Insurance Fraud.**

This project was a huge eye-opener for me. It wasn't just about math; it was about understanding how people try to cheat the system and how we can use data to protect honest customers.

## The Challenge: "The 0.003% Problem"
Insurance fraud is like searching for a needle in a haystack. In my dataset of nearly **10,000 records**, only **32 cases** were actual fraud. That’s a **0.003% fraud rate**. 

Standard models usually fail here because they simply ignore the fraud cases as "errors." My main goal was to find a way to make the models pay attention to these rare but critical events.



## How I tackled the analysis

Instead of just following a textbook, I experimented with different strategies to see what worked for the specific Greek market data:

* **Looking for the "Red Flags":** I started with an Exploratory Data Analysis (EDA) and found that location matters—Attica had a disproportionately high number of suspicious claims.
* **Balancing the Scales:** To fix the extreme imbalance, I used **SMOTE** and **ROSE** techniques. These methods "synthetically" increased the fraud cases so the algorithms could actually learn their patterns.
* **The Algorithm Showdown:** I didn't settle for one model. I built and compared **Logistic Regression**, **Random Forest**, and **XGBoost**. I also used **t-SNE** to visualize how fraud cases cluster together in a multidimensional space.
* **The Result:** The **Random Forest** model emerged as the winner, achieving an **AUC score of 0.697**. While fraud detection is never 100% perfect, this provided a solid framework for identifying high-risk claims.


---
**Konstantinos Pasapnois** *Statistics Student @ AUEB*
